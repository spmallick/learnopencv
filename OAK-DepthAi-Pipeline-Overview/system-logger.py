import depthai as dai

def printSystemInformation(info):
    m = 1024 * 1024 # MiB
    print(f"Ddr used / total - {info.ddrMemoryUsage.used / m:.2f} / {info.ddrMemoryUsage.total / m:.2f} MiB")
    print(f"Cmx used / total - {info.cmxMemoryUsage.used / m:.2f} / {info.cmxMemoryUsage.total / m:.2f} MiB")
    print(f"LeonCss heap used / total - {info.leonCssMemoryUsage.used / m:.2f} / {info.leonCssMemoryUsage.total / m:.2f} MiB")
    print(f"LeonMss heap used / total - {info.leonMssMemoryUsage.used / m:.2f} / {info.leonMssMemoryUsage.total / m:.2f} MiB")
    t = info.chipTemperature
    print(f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa: {t.upa:.2f}, dss: {t.dss:.2f}")
    print(f"Cpu usage - Leon CSS: {info.leonCssCpuUsage.average * 100:.2f}%, Leon MSS: {info.leonMssCpuUsage.average * 100:.2f} %")
    print("----------------------------------------")

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
sysLog = pipeline.create(dai.node.SystemLogger)
linkOut = pipeline.create(dai.node.XLinkOut)

linkOut.setStreamName("sysinfo")

# Properties
sysLog.setRate(1)  # 1 Hz

# Linking
sysLog.out.link(linkOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the system info
    qSysInfo = device.getOutputQueue(name="sysinfo", maxSize=4, blocking=False)

    while True:
        sysInfo = qSysInfo.get()  # Blocking call, will wait until a new data has arrived
        printSystemInformation(sysInfo)