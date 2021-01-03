//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.core;



// C++: class Algorithm
/**
 * This is a base class for all more or less complex algorithms in OpenCV
 *
 * especially for classes of algorithms, for which there can be multiple implementations. The examples
 * are stereo correspondence (for which there are algorithms like block matching, semi-global block
 * matching, graph-cut etc.), background subtraction (which can be done using mixture-of-gaussians
 * models, codebook-based algorithm etc.), optical flow (block matching, Lucas-Kanade, Horn-Schunck
 * etc.).
 *
 * Here is example of SimpleBlobDetector use in your application via Algorithm interface:
 * SNIPPET: snippets/core_various.cpp Algorithm
 */
public class Algorithm {

    protected final long nativeObj;
    protected Algorithm(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static Algorithm __fromPtr__(long addr) { return new Algorithm(addr); }

    //
    // C++:  String cv::Algorithm::getDefaultName()
    //

    /**
     * Returns the algorithm string identifier.
     * This string is used as top level xml/yml node tag when the object is saved to a file or string.
     * @return automatically generated
     */
    public String getDefaultName() {
        return getDefaultName_0(nativeObj);
    }


    //
    // C++:  bool cv::Algorithm::empty()
    //

    /**
     * Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read
     * @return automatically generated
     */
    public boolean empty() {
        return empty_0(nativeObj);
    }


    //
    // C++:  void cv::Algorithm::clear()
    //

    /**
     * Clears the algorithm state
     */
    public void clear() {
        clear_0(nativeObj);
    }


    //
    // C++:  void cv::Algorithm::read(FileNode fn)
    //

    // Unknown type 'FileNode' (I), skipping the function


    //
    // C++:  void cv::Algorithm::save(String filename)
    //

    /**
     * Saves the algorithm to a file.
     * In order to make this method work, the derived class must implement Algorithm::write(FileStorage&amp; fs).
     * @param filename automatically generated
     */
    public void save(String filename) {
        save_0(nativeObj, filename);
    }


    //
    // C++:  void cv::Algorithm::write(Ptr_FileStorage fs, String name = String())
    //

    // Unknown type 'Ptr_FileStorage' (I), skipping the function


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  String cv::Algorithm::getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  bool cv::Algorithm::empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  void cv::Algorithm::clear()
    private static native void clear_0(long nativeObj);

    // C++:  void cv::Algorithm::save(String filename)
    private static native void save_0(long nativeObj, String filename);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
