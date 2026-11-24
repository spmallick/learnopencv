/* CUDA 12.9 CUPTI probe for inspect_cuda.py. Runtime callbacks count API
 * submissions; memcpy activity records report actual transfers by direction. */
#include <cupti.h>
#include <cuda_runtime_api.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Keep this order consistent with the names in inspect_cuda.py. */
enum {
    CAPTURES, GRAPH_LAUNCHES, KERNEL_LAUNCHES, H2D_COPIES, H2D_BYTES,
    MALLOCS, MALLOC_BYTES, FREES, NCOUNTERS
};
static _Atomic uint64_t counts[NCOUNTERS];
static _Atomic int incomplete;
static CUpti_SubscriberHandle subscriber;

static void CUPTIAPI observe(void *user, CUpti_CallbackDomain domain,
                             CUpti_CallbackId id, const void *record)
{
    (void)user;
    const CUpti_CallbackData *call = record;
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || call->callbackSite != CUPTI_API_EXIT)
        return;

    /* Include default-stream variants. Only matched functions return cudaError_t. */
    const char *name = call->functionName;
    unsigned counter;
    if (strncmp(name, "cudaStreamBeginCapture", strlen("cudaStreamBeginCapture")) == 0)
        counter = CAPTURES;
    else if (strncmp(name, "cudaGraphLaunch", strlen("cudaGraphLaunch")) == 0)
        counter = GRAPH_LAUNCHES;
    else if (strncmp(name, "cudaLaunchKernel", strlen("cudaLaunchKernel")) == 0)
        counter = KERNEL_LAUNCHES;
    else if (id == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020)
        counter = MALLOCS;
    else if (id == CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020)
        counter = FREES;
    else
        return;

    if (!call->functionReturnValue || *(const cudaError_t *)call->functionReturnValue != cudaSuccess)
        return;
    atomic_fetch_add(&counts[counter], 1);
    if (counter == MALLOCS) {
        const cudaMalloc_v3020_params *args = call->functionParams;
        atomic_fetch_add(&counts[MALLOC_BYTES], args->size);
    }
}

static void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *max_records)
{
    /* CUPTI owns this buffer until it calls consume_buffer. */
    *size = 1u << 20;
    *buffer = malloc(*size);
    *max_records = 0;
    if (!*buffer) {
        *size = 0;
        atomic_store(&incomplete, 1);
    }
}

static void CUPTIAPI consume_buffer(CUcontext context, uint32_t stream, uint8_t *buffer,
                                    size_t size, size_t valid)
{
    (void)size;
    CUpti_Activity *record = NULL;
    CUptiResult rc;
    while ((rc = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            /* Includes transfers for which cudaMemcpyDefault inferred the direction. */
            const CUpti_ActivityMemcpy6 *copy = (const CUpti_ActivityMemcpy6 *)record;
            if (copy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD) {
                atomic_fetch_add(&counts[H2D_COPIES], 1);
                atomic_fetch_add(&counts[H2D_BYTES], copy->bytes);
            }
        }
    }
    /* MAX_LIMIT_REACHED is the documented end-of-buffer status. */
    size_t dropped = 0;
    if (rc != CUPTI_ERROR_MAX_LIMIT_REACHED ||
        cuptiActivityGetNumDroppedRecords(context, stream, &dropped) != CUPTI_SUCCESS || dropped)
        atomic_store(&incomplete, 1);
    free(buffer);
}

int trace_start(void)
{
    CUptiResult rc = cuptiSubscribe(&subscriber, observe, NULL);
    if (rc == CUPTI_SUCCESS)
        rc = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    if (rc == CUPTI_SUCCESS)
        rc = cuptiActivityRegisterCallbacks(request_buffer, consume_buffer);
    if (rc == CUPTI_SUCCESS)
        rc = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    if (rc != CUPTI_SUCCESS)
        fprintf(stderr, "CUPTI initialization failed with error %d\n", (int)rc);
    return (int)rc;
}

int trace_flush(void)
{
    /* The host-visible forward has finished. Deliver its pending memcpy records
     * before Python samples the counters, and reject any incomplete trace. */
    return cuptiActivityFlushAll(0) != CUPTI_SUCCESS || atomic_load(&incomplete);
}

uint64_t trace_read(unsigned index)
{
    return index < NCOUNTERS ? atomic_load(&counts[index]) : 0;
}
