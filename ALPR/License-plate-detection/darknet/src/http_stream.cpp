#define _XOPEN_SOURCE
#include "image.h"
#include "http_stream.h"

//
// a single-threaded, multi client(using select), debug webserver - streaming out mjpg.
//  on win, _WIN32 has to be defined, must link against ws2_32.lib (socks on linux are for free)
//

#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <ctime>
using std::cerr;
using std::endl;

//
// socket related abstractions:
//
#ifdef _WIN32
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "ws2_32.lib")
#endif
#define WIN32_LEAN_AND_MEAN
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "gettimeofday.h"
#define PORT        unsigned long
#define ADDRPOINTER   int*
struct _INIT_W32DATA
{
    WSADATA w;
    _INIT_W32DATA() { WSAStartup(MAKEWORD(2, 1), &w); }
} _init_once;

// Graceful closes will first close their output channels and then wait for the peer
// on the other side of the connection to close its output channels. When both sides are done telling
// each other they won,t be sending any more data (i.e., closing output channels),
// the connection can be closed fully, with no risk of reset.
static int close_socket(SOCKET s) {
    int close_output = ::shutdown(s, 1); // 0 close input, 1 close output, 2 close both
    char *buf = (char *)calloc(1024, sizeof(char));
    ::recv(s, buf, 1024, 0);
    free(buf);
    int close_input = ::shutdown(s, 0);
    int result = ::closesocket(s);
    cerr << "Close socket: out = " << close_output << ", in = " << close_input << " \n";
    return result;
}
#else   // _WIN32 - else: nix
#include "darkunistd.h"
#include <fcntl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#define PORT        unsigned short
#define SOCKET    int
#define HOSTENT  struct hostent
#define SOCKADDR    struct sockaddr
#define SOCKADDR_IN  struct sockaddr_in
#define ADDRPOINTER  unsigned int*
#ifndef INVALID_SOCKET
#define INVALID_SOCKET -1
#endif
#ifndef SOCKET_ERROR
#define SOCKET_ERROR   -1
#endif
struct _IGNORE_PIPE_SIGNAL
{
    struct sigaction new_actn, old_actn;
    _IGNORE_PIPE_SIGNAL() {
        new_actn.sa_handler = SIG_IGN;  // ignore the broken pipe signal
        sigemptyset(&new_actn.sa_mask);
        new_actn.sa_flags = 0;
        sigaction(SIGPIPE, &new_actn, &old_actn);
        // sigaction (SIGPIPE, &old_actn, NULL); // - to restore the previous signal handling
    }
} _init_once;

static int close_socket(SOCKET s) {
    int close_output = ::shutdown(s, 1); // 0 close input, 1 close output, 2 close both
    char *buf = (char *)calloc(1024, sizeof(char));
    ::recv(s, buf, 1024, 0);
    free(buf);
    int close_input = ::shutdown(s, 0);
    int result = close(s);
    std::cerr << "Close socket: out = " << close_output << ", in = " << close_input << " \n";
    return result;
}
#endif // _WIN32


class JSON_sender
{
    SOCKET sock;
    SOCKET maxfd;
    fd_set master;
    int timeout; // master sock timeout, shutdown after timeout usec.
    int close_all_sockets;

    int _write(int sock, char const*const s, int len)
    {
        if (len < 1) { len = strlen(s); }
        return ::send(sock, s, len, 0);
    }

public:

    JSON_sender(int port = 0, int _timeout = 400000)
        : sock(INVALID_SOCKET)
        , timeout(_timeout)
    {
        close_all_sockets = 0;
        FD_ZERO(&master);
        if (port)
            open(port);
    }

    ~JSON_sender()
    {
        close_all();
        release();
    }

    bool release()
    {
        if (sock != INVALID_SOCKET)
            ::shutdown(sock, 2);
        sock = (INVALID_SOCKET);
        return false;
    }

    void close_all()
    {
        close_all_sockets = 1;
        write("\n]");   // close JSON array
    }

    bool open(int port)
    {
        sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        SOCKADDR_IN address;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family = AF_INET;
        address.sin_port = htons(port);    // ::htons(port);
        int reuse = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEADDR) failed" << endl;

        // Non-blocking sockets
        // Windows: ioctlsocket() and FIONBIO
        // Linux: fcntl() and O_NONBLOCK
#ifdef WIN32
        unsigned long i_mode = 1;
        int result = ioctlsocket(sock, FIONBIO, &i_mode);
        if (result != NO_ERROR) {
            std::cerr << "ioctlsocket(FIONBIO) failed with error: " << result << std::endl;
        }
#else // WIN32
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif // WIN32

#ifdef SO_REUSEPORT
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEPORT) failed" << endl;
#endif
        if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR)
        {
            cerr << "error JSON_sender: couldn't bind sock " << sock << " to port " << port << "!" << endl;
            return release();
        }
        if (::listen(sock, 10) == SOCKET_ERROR)
        {
            cerr << "error JSON_sender: couldn't listen on sock " << sock << " on port " << port << " !" << endl;
            return release();
        }
        FD_ZERO(&master);
        FD_SET(sock, &master);
        maxfd = sock;
        return true;
    }

    bool isOpened()
    {
        return sock != INVALID_SOCKET;
    }

    bool write(char const* outputbuf)
    {
        fd_set rread = master;
        struct timeval select_timeout = { 0, 0 };
        struct timeval socket_timeout = { 0, timeout };
        if (::select(maxfd + 1, &rread, NULL, NULL, &select_timeout) <= 0)
            return true; // nothing broken, there's just noone listening

        int outlen = static_cast<int>(strlen(outputbuf));

#ifdef _WIN32
        for (unsigned i = 0; i<rread.fd_count; i++)
        {
            int addrlen = sizeof(SOCKADDR);
            SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
#else
        for (int s = 0; s <= maxfd; s++)
        {
            socklen_t addrlen = sizeof(SOCKADDR);
            if (!FD_ISSET(s, &rread))      // ... on linux it's a bitmask ;)
                continue;
#endif
            if (s == sock) // request on master socket, accept and send main header.
            {
                SOCKADDR_IN address = { 0 };
                SOCKET      client = ::accept(sock, (SOCKADDR*)&address, &addrlen);
                if (client == SOCKET_ERROR)
                {
                    cerr << "error JSON_sender: couldn't accept connection on sock " << sock << " !" << endl;
                    return false;
                }
                if (setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error JSON_sender: SO_RCVTIMEO setsockopt failed\n";
                }
                if (setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error JSON_sender: SO_SNDTIMEO setsockopt failed\n";
                }
                maxfd = (maxfd>client ? maxfd : client);
                FD_SET(client, &master);
                _write(client, "HTTP/1.0 200 OK\r\n", 0);
                _write(client,
                    "Server: Mozarella/2.2\r\n"
                    "Accept-Range: bytes\r\n"
                    "Connection: close\r\n"
                    "Max-Age: 0\r\n"
                    "Expires: 0\r\n"
                    "Cache-Control: no-cache, private\r\n"
                    "Pragma: no-cache\r\n"
                    "Content-Type: application/json\r\n"
                    //"Content-Type: multipart/x-mixed-replace; boundary=boundary\r\n"
                    "\r\n", 0);
                _write(client, "[\n", 0);   // open JSON array
                int n = _write(client, outputbuf, outlen);
                cerr << "JSON_sender: new client " << client << endl;
            }
            else // existing client, just stream pix
            {
                //char head[400];
                // application/x-resource+json or application/x-collection+json -  when you are representing REST resources and collections
                // application/json or text/json or text/javascript or text/plain.
                // https://stackoverflow.com/questions/477816/what-is-the-correct-json-content-type
                //sprintf(head, "\r\nContent-Length: %zu\r\n\r\n", outlen);
                //sprintf(head, "--boundary\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", outlen);
                //_write(s, head, 0);
                if (!close_all_sockets) _write(s, ", \n", 0);
                int n = _write(s, outputbuf, outlen);
                if (n < (int)outlen)
                {
                    cerr << "JSON_sender: kill client " << s << endl;
                    close_socket(s);
                    //::shutdown(s, 2);
                    FD_CLR(s, &master);
                }

                if (close_all_sockets) {
                    int result = close_socket(s);
                    cerr << "JSON_sender: close clinet: " << result << " \n";
                    continue;
                }
            }
        }
        if (close_all_sockets) {
            int result = close_socket(sock);
            cerr << "JSON_sender: close acceptor: " << result << " \n\n";
        }
        return true;
        }
};
// ----------------------------------------

static std::unique_ptr<JSON_sender> js_ptr;
static std::mutex mtx;

void delete_json_sender()
{
    std::lock_guard<std::mutex> lock(mtx);
    js_ptr.release();
}

void send_json_custom(char const* send_buf, int port, int timeout)
{
    try {
        std::lock_guard<std::mutex> lock(mtx);
        if(!js_ptr) js_ptr.reset(new JSON_sender(port, timeout));

        js_ptr->write(send_buf);
    }
    catch (...) {
        cerr << " Error in send_json_custom() function \n";
    }
}

void send_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, int port, int timeout)
{
    try {
        char *send_buf = detection_to_json(dets, nboxes, classes, names, frame_id, NULL);

        send_json_custom(send_buf, port, timeout);
        std::cout << " JSON-stream sent. \n";

        free(send_buf);
    }
    catch (...) {
        cerr << " Error in send_json() function \n";
    }
}
// ----------------------------------------


#ifdef OPENCV

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio.hpp>
#endif
using namespace cv;



class MJPG_sender
{
    SOCKET sock;
    SOCKET maxfd;
    fd_set master;
    int timeout; // master sock timeout, shutdown after timeout usec.
    int quality; // jpeg compression [1..100]
    int close_all_sockets;

    int _write(int sock, char const*const s, int len)
    {
        if (len < 1) { len = strlen(s); }
        return ::send(sock, s, len, 0);
    }

public:

    MJPG_sender(int port = 0, int _timeout = 400000, int _quality = 30)
        : sock(INVALID_SOCKET)
        , timeout(_timeout)
        , quality(_quality)
    {
        close_all_sockets = 0;
        FD_ZERO(&master);
        if (port)
            open(port);
    }

    ~MJPG_sender()
    {
        close_all();
        release();
    }

    bool release()
    {
        if (sock != INVALID_SOCKET)
            ::shutdown(sock, 2);
        sock = (INVALID_SOCKET);
        return false;
    }

    void close_all()
    {
        close_all_sockets = 1;
        cv::Mat tmp(cv::Size(10, 10), CV_8UC3);
        write(tmp);
    }

    bool open(int port)
    {
        sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        SOCKADDR_IN address;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family = AF_INET;
        address.sin_port = htons(port);    // ::htons(port);
        int reuse = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEADDR) failed" << endl;

        // Non-blocking sockets
        // Windows: ioctlsocket() and FIONBIO
        // Linux: fcntl() and O_NONBLOCK
#ifdef WIN32
        unsigned long i_mode = 1;
        int result = ioctlsocket(sock, FIONBIO, &i_mode);
        if (result != NO_ERROR) {
            std::cerr << "ioctlsocket(FIONBIO) failed with error: " << result << std::endl;
        }
#else // WIN32
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif // WIN32

#ifdef SO_REUSEPORT
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse)) < 0)
            cerr << "setsockopt(SO_REUSEPORT) failed" << endl;
#endif
        if (::bind(sock, (SOCKADDR*)&address, sizeof(SOCKADDR_IN)) == SOCKET_ERROR)
        {
            cerr << "error MJPG_sender: couldn't bind sock " << sock << " to port " << port << "!" << endl;
            return release();
        }
        if (::listen(sock, 10) == SOCKET_ERROR)
        {
            cerr << "error MJPG_sender: couldn't listen on sock " << sock << " on port " << port << " !" << endl;
            return release();
        }
        FD_ZERO(&master);
        FD_SET(sock, &master);
        maxfd = sock;
        return true;
    }

    bool isOpened()
    {
        return sock != INVALID_SOCKET;
    }

    bool write(const Mat & frame)
    {
        fd_set rread = master;
        struct timeval select_timeout = { 0, 0 };
        struct timeval socket_timeout = { 0, timeout };
        if (::select(maxfd + 1, &rread, NULL, NULL, &select_timeout) <= 0)
            return true; // nothing broken, there's just noone listening

        std::vector<uchar> outbuf;
        std::vector<int> params;
        params.push_back(IMWRITE_JPEG_QUALITY);
        params.push_back(quality);
        cv::imencode(".jpg", frame, outbuf, params);  //REMOVED FOR COMPATIBILITY
        // https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac
        //std::cerr << "cv::imencode call disabled!" << std::endl;
        int outlen = static_cast<int>(outbuf.size());

#ifdef _WIN32
        for (unsigned i = 0; i<rread.fd_count; i++)
        {
            int addrlen = sizeof(SOCKADDR);
            SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
#else
        for (int s = 0; s <= maxfd; s++)
        {
            socklen_t addrlen = sizeof(SOCKADDR);
            if (!FD_ISSET(s, &rread))      // ... on linux it's a bitmask ;)
                continue;
#endif
            if (s == sock) // request on master socket, accept and send main header.
            {
                SOCKADDR_IN address = { 0 };
                SOCKET      client = ::accept(sock, (SOCKADDR*)&address, &addrlen);
                if (client == SOCKET_ERROR)
                {
                    cerr << "error MJPG_sender: couldn't accept connection on sock " << sock << " !" << endl;
                    return false;
                }
                if (setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error MJPG_sender: SO_RCVTIMEO setsockopt failed\n";
                }
                if (setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (char *)&socket_timeout, sizeof(socket_timeout)) < 0) {
                    cerr << "error MJPG_sender: SO_SNDTIMEO setsockopt failed\n";
                }
                maxfd = (maxfd>client ? maxfd : client);
                FD_SET(client, &master);
                _write(client, "HTTP/1.0 200 OK\r\n", 0);
                _write(client,
                    "Server: Mozarella/2.2\r\n"
                    "Accept-Range: bytes\r\n"
                    "Connection: close\r\n"
                    "Max-Age: 0\r\n"
                    "Expires: 0\r\n"
                    "Cache-Control: no-cache, private\r\n"
                    "Pragma: no-cache\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
                    "\r\n", 0);
                cerr << "MJPG_sender: new client " << client << endl;
            }
            else // existing client, just stream pix
            {
                if (close_all_sockets) {
                    int result = close_socket(s);
                    cerr << "MJPG_sender: close clinet: " << result << " \n";
                    continue;
                }

                char head[400];
                sprintf(head, "--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", outlen);
                _write(s, head, 0);
                int n = _write(s, (char*)(&outbuf[0]), outlen);
                cerr << "known client: " << s << ", sent = " << n << ", must be sent outlen = " << outlen << endl;
                if (n < (int)outlen)
                {
                    cerr << "MJPG_sender: kill client " << s << endl;
                    //::shutdown(s, 2);
                    close_socket(s);
                    FD_CLR(s, &master);
                }
            }
        }
        if (close_all_sockets) {
            int result = close_socket(sock);
            cerr << "MJPG_sender: close acceptor: " << result << " \n\n";
        }
        return true;
    }
};
// ----------------------------------------

static std::mutex mtx_mjpeg;

//struct mat_cv : cv::Mat { int a[0]; };

void send_mjpeg(mat_cv* mat, int port, int timeout, int quality)
{
    try {
        std::lock_guard<std::mutex> lock(mtx_mjpeg);
        static MJPG_sender wri(port, timeout, quality);
        //cv::Mat mat = cv::cvarrToMat(ipl);
        wri.write(*(cv::Mat*)mat);
        std::cout << " MJPEG-stream sent. \n";
    }
    catch (...) {
        cerr << " Error in send_mjpeg() function \n";
    }
}
// ----------------------------------------

std::string get_system_frame_time_string()
{
    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    struct tm *tmp_buf = localtime(&t);
    char buff[256];
    std::strftime(buff, 256, "%A %F %T", tmp_buf);
    std::string system_frame_time = buff;
    return system_frame_time;
}
// ----------------------------------------


#ifdef __CYGWIN__
int send_http_post_request(char *http_post_host, int server_port, const char *videosource,
    detection *dets, int nboxes, int classes, char **names, long long int frame_id, int ext_output, int timeout)
{
    std::cerr << " send_http_post_request() isn't implemented \n";
    return 0;
}
#else   //  __CYGWIN__

#ifndef   NI_MAXHOST
#define   NI_MAXHOST 1025
#endif

#ifndef   NI_NUMERICHOST
#define NI_NUMERICHOST  0x02
#endif

//#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

// https://webhook.site/
// https://github.com/yhirose/cpp-httplib
// sent POST http request
int send_http_post_request(char *http_post_host, int server_port, const char *videosource,
    detection *dets, int nboxes, int classes, char **names, long long int frame_id, int ext_output, int timeout)
{
    const float thresh = 0.005; // function get_network_boxes() has already filtred dets by actual threshold

    std::string message;

    for (int i = 0; i < nboxes; ++i) {
        char labelstr[4096] = { 0 };
        int class_id = -1;
        for (int j = 0; j < classes; ++j) {
            int show = strncmp(names[j], "dont_show", 9);
            if (dets[i].prob[j] > thresh && show) {
                if (class_id < 0) {
                    strcat(labelstr, names[j]);
                    class_id = j;
                    char buff[10];
                    sprintf(buff, " (%2.0f%%)", dets[i].prob[j] * 100);
                    strcat(labelstr, buff);
                }
                else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%% ", names[j], dets[i].prob[j] * 100);
            }
        }
        if (class_id >= 0) {
            message += std::string(names[class_id]) + std::string(", id: ") + std::to_string(class_id) + "\n";
        }
    }

    if (!message.empty())
    {
        std::string time = get_system_frame_time_string();
        message += "\ntime:\n" + time + "\n";
        message += "videosource:\n" + std::string(videosource);

        std::string http_post_host_str = http_post_host;
        int slash_index = http_post_host_str.find("/");

        std::string http_path = http_post_host_str.substr(slash_index, http_post_host_str.length() - slash_index);
        http_post_host_str = http_post_host_str.substr(0, slash_index);

        // send HTTP-Post request
        httplib::Client cli(http_post_host_str.c_str(), server_port, timeout);
        auto res = cli.Post(http_path.c_str(), message, "text/plain");

        return 1;
    }

    return 0;
}
#endif   //  __CYGWIN__

#endif      // OPENCV

// -----------------------------------------------------

#if __cplusplus >= 201103L || _MSC_VER >= 1900  // C++11

#include <chrono>
#include <iostream>

static std::chrono::steady_clock::time_point steady_start, steady_end;
static double total_time;

double get_time_point() {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    //uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count();
    return std::chrono::duration_cast<std::chrono::microseconds>(current_time.time_since_epoch()).count();
}

void start_timer() {
    steady_start = std::chrono::steady_clock::now();
}

void stop_timer() {
    steady_end = std::chrono::steady_clock::now();
}

double get_time() {
    double took_time = std::chrono::duration<double>(steady_end - steady_start).count();
    total_time += took_time;
    return took_time;
}

void stop_timer_and_show() {
    stop_timer();
    std::cout << " " << get_time() * 1000 << " msec" << std::endl;
}

void stop_timer_and_show_name(char *name) {
    stop_timer();
    std::cout << " " << name;
    std::cout << " " << get_time() * 1000 << " msec" << std::endl;
}

void show_total_time() {
    std::cout << " Total: " << total_time * 1000 << " msec" << std::endl;
}


int custom_create_thread(custom_thread_t * tid, const custom_attr_t * attr, void *(*func) (void *), void *arg)
{
    std::thread *ptr = new std::thread(func, arg);
    *tid = (custom_thread_t *)ptr;
    if (tid) return 0;
    else return -1;
}

int custom_join(custom_thread_t tid, void **value_ptr)
{
    std::thread *ptr = (std::thread *)tid;
    if (ptr) {
        ptr->join();
        delete ptr;
        return 0;
    }
    else printf(" Error: ptr of thread is NULL in custom_join() \n");

    return -1;
}

int custom_atomic_load_int(volatile int* obj)
{
    const volatile std::atomic<int>* ptr_a = (const volatile std::atomic<int>*)obj;
    return std::atomic_load(ptr_a);
}

void custom_atomic_store_int(volatile int* obj, int desr)
{
    volatile std::atomic<int>* ptr_a = (volatile std::atomic<int>*)obj;
    std::atomic_store(ptr_a, desr);
}

int get_num_threads()
{
    return std::thread::hardware_concurrency();
}

#if !defined(__MINGW64__)
void this_thread_sleep_for(int ms_time)
{
    std::chrono::milliseconds dura(ms_time);
    std::this_thread::sleep_for(dura);
}
#else
void this_thread_sleep_for(int ms_time)
{
    std::cerr << " this_thread_sleep_for() isn't implemented \n";
    return;
}
#endif

void this_thread_yield()
{
    std::this_thread::yield();
}

#else // C++11
#include <iostream>

double get_time_point() { return 0; }
void start_timer() {}
void stop_timer() {}
double get_time() { return 0; }
void stop_timer_and_show() {
    std::cout << " stop_timer_and_show() isn't implemented " << std::endl;
}
void stop_timer_and_show_name(char *name) { stop_timer_and_show(); }
void total_time() {}
#endif // C++11

#include <deque>
#include <vector>
#include <iostream>
#include "blas.h"
#include "utils.h"

struct similarity_detections_t {
    int old_id, new_id;
    float sim;
};

int check_prob(detection det, float thresh)
{
    for (int i = 0; i < det.classes; ++i) {
        if (det.prob[i] > thresh) return 1;
    }
    return 0;
}

int check_classes_id(detection det1, detection det2, float thresh)
{
    if (det1.classes != det2.classes) {
        printf(" Error: det1.classes != det2.classes \n");
        getchar();
    }

    int det1_id = -1;
    float det1_prob = 0;
    int det2_id = -1;
    float det2_prob = 0;

    for (int i = 0; i < det1.classes; ++i) {
        if (det1.prob[i] > thresh && det1.prob[i] > det1_prob) {
            det1_prob = det1.prob[i];
            det1_id = i;
        }
        if (det2.prob[i] > thresh && det2.prob[i] > det2_prob) {
            det2_prob = det2.prob[i];
            det2_id = i;
        }
    }

    if (det1_id == det2_id && det2_id != -1) return 1;

    //for (int i = 0; i < det1.classes; ++i) {
    //    if (det1.prob[i] > thresh && det2.prob[i] > thresh) return 1;
    //}
    return 0;
}

int fill_remaining_id(detection *new_dets, int new_dets_num, int new_track_id, float thresh, int detection_count)
{
    for (int i = 0; i < new_dets_num; ++i) {
        if (new_dets[i].track_id == 0 && check_prob(new_dets[i], thresh)) {
            //printf(" old_tid = %d, new_tid = %d, sim = %f \n", new_dets[i].track_id, new_track_id, new_dets[i].sim);
            if (new_dets[i].sort_class > detection_count) {
                new_dets[i].track_id = new_track_id;
                new_track_id++;
            }
        }
    }
    return new_track_id;
}

float *make_float_array(float* src, size_t size)
{
    float *dst = (float*)xcalloc(size, sizeof(float));
    memcpy(dst, src, size*sizeof(float));
    return dst;
}

struct detection_t : detection {
    int det_count;
    detection_t(detection det) : detection(det), det_count(0)
    {
        if (embeddings) embeddings = make_float_array(det.embeddings, embedding_size);
        if (prob) prob = make_float_array(det.prob, classes);
        if (uc) uc = make_float_array(det.uc, 4);
    }

    detection_t(detection_t const& det) : detection(det)
    {
        if (embeddings) embeddings = make_float_array(det.embeddings, embedding_size);
        if (prob) prob = make_float_array(det.prob, classes);
        if (uc) uc = make_float_array(det.uc, 4);
    }

    ~detection_t() {
        if (embeddings) free(embeddings);
        if (prob) free(prob);
        if (uc) free(uc);
    }
};



void set_track_id(detection *new_dets, int new_dets_num, float thresh, float sim_thresh, float track_ciou_norm, int deque_size, int dets_for_track, int dets_for_show)
{
    static int new_track_id = 1;
    static std::deque<std::vector<detection_t>> old_dets_dq;

    // copy detections from queue of vectors to the one vector
    std::vector<detection_t> old_dets;
    for (std::vector<detection_t> &v : old_dets_dq) {
        for (int i = 0; i < v.size(); ++i) {
            old_dets.push_back(v[i]);
        }
    }

    std::vector<similarity_detections_t> sim_det(old_dets.size() * new_dets_num);

    // calculate similarity
    for (int old_id = 0; old_id < old_dets.size(); ++old_id) {
        for (int new_id = 0; new_id < new_dets_num; ++new_id) {
            const int index = old_id*new_dets_num + new_id;
            const float sim = cosine_similarity(new_dets[new_id].embeddings, old_dets[old_id].embeddings, old_dets[0].embedding_size);
            sim_det[index].new_id = new_id;
            sim_det[index].old_id = old_id;
            sim_det[index].sim = sim;
        }
    }

    // sort similarity
    std::sort(sim_det.begin(), sim_det.end(), [](similarity_detections_t v1, similarity_detections_t v2) { return v1.sim > v2.sim; });
    //if(sim_det.size() > 0) printf(" sim_det_first = %f, sim_det_end = %f \n", sim_det.begin()->sim, sim_det.rbegin()->sim);

    std::vector<int> new_idx(new_dets_num, 1);
    std::vector<int> old_idx(old_dets.size(), 1);
    std::vector<int> track_idx(new_track_id, 1);

    // match objects
    for (int index = 0; index < new_dets_num*old_dets.size(); ++index) {
        const int new_id = sim_det[index].new_id;
        const int old_id = sim_det[index].old_id;
        const int track_id = old_dets[old_id].track_id;
        const int det_count = old_dets[old_id].sort_class;
        //printf(" ciou = %f \n", box_ciou(new_dets[new_id].bbox, old_dets[old_id].bbox));
        if (track_idx[track_id] && new_idx[new_id] && old_idx[old_id] && check_classes_id(new_dets[new_id], old_dets[old_id], thresh)) {
            float sim = sim_det[index].sim;
            //float ciou = box_ciou(new_dets[new_id].bbox, old_dets[old_id].bbox);
            float ciou = box_iou(new_dets[new_id].bbox, old_dets[old_id].bbox);
            sim = sim * (1 - track_ciou_norm) + ciou * track_ciou_norm;
            if (sim_thresh < sim && new_dets[new_id].sim < sim) {
                new_dets[new_id].sim = sim;
                new_dets[new_id].track_id = track_id;
                new_dets[new_id].sort_class = det_count + 1;
                //new_idx[new_id] = 0;
                old_idx[old_id] = 0;
                if(track_id) track_idx[track_id] = 0;
            }
        }
    }

    // set new track_id
    new_track_id = fill_remaining_id(new_dets, new_dets_num, new_track_id, thresh, dets_for_track);

    // store new_detections to the queue of vectors
    std::vector<detection_t> new_det_vec;
    for (int i = 0; i < new_dets_num; ++i) {
        if (check_prob(new_dets[i], thresh)) {
            new_det_vec.push_back(new_dets[i]);
        }
    }

    // add new
    old_dets_dq.push_back(new_det_vec);
    // remove old
    if (old_dets_dq.size() > deque_size) old_dets_dq.pop_front();

    // remove detection which were detected only on few frames
    for (int i = 0; i < new_dets_num; ++i) {
        if (new_dets[i].sort_class < dets_for_show) {
            for (int j = 0; j < new_dets[i].classes; ++j) {
                new_dets[i].prob[j] = 0;
            }
        }
    }
}
