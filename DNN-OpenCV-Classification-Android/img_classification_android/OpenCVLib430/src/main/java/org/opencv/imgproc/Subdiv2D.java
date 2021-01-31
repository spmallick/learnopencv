//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.imgproc;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat4;
import org.opencv.core.MatOfFloat6;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.utils.Converters;

// C++: class Subdiv2D

public class Subdiv2D {

    protected final long nativeObj;
    protected Subdiv2D(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static Subdiv2D __fromPtr__(long addr) { return new Subdiv2D(addr); }

    // C++: enum <unnamed>
    public static final int
            PTLOC_ERROR = -2,
            PTLOC_OUTSIDE_RECT = -1,
            PTLOC_INSIDE = 0,
            PTLOC_VERTEX = 1,
            PTLOC_ON_EDGE = 2,
            NEXT_AROUND_ORG = 0x00,
            NEXT_AROUND_DST = 0x22,
            PREV_AROUND_ORG = 0x11,
            PREV_AROUND_DST = 0x33,
            NEXT_AROUND_LEFT = 0x13,
            NEXT_AROUND_RIGHT = 0x31,
            PREV_AROUND_LEFT = 0x20,
            PREV_AROUND_RIGHT = 0x02;


    //
    // C++:   cv::Subdiv2D::Subdiv2D(Rect rect)
    //

    /**
     *
     *
     *     @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
     *
     *     The function creates an empty Delaunay subdivision where 2D points can be added using the function
     *     insert() . All of the points to be added must be within the specified rectangle, otherwise a runtime
     *     error is raised.
     */
    public Subdiv2D(Rect rect) {
        nativeObj = Subdiv2D_0(rect.x, rect.y, rect.width, rect.height);
    }


    //
    // C++:   cv::Subdiv2D::Subdiv2D()
    //

    /**
     * creates an empty Subdiv2D object.
     *     To create a new empty Delaunay subdivision you need to use the #initDelaunay function.
     */
    public Subdiv2D() {
        nativeObj = Subdiv2D_1();
    }


    //
    // C++:  Point2f cv::Subdiv2D::getVertex(int vertex, int* firstEdge = 0)
    //

    /**
     * Returns vertex location from vertex ID.
     *
     *     @param vertex vertex ID.
     *     @param firstEdge Optional. The first edge ID which is connected to the vertex.
     *     @return vertex (x,y)
     */
    public Point getVertex(int vertex, int[] firstEdge) {
        double[] firstEdge_out = new double[1];
        Point retVal = new Point(getVertex_0(nativeObj, vertex, firstEdge_out));
        if(firstEdge!=null) firstEdge[0] = (int)firstEdge_out[0];
        return retVal;
    }

    /**
     * Returns vertex location from vertex ID.
     *
     *     @param vertex vertex ID.
     *     @return vertex (x,y)
     */
    public Point getVertex(int vertex) {
        return new Point(getVertex_1(nativeObj, vertex));
    }


    //
    // C++:  int cv::Subdiv2D::edgeDst(int edge, Point2f* dstpt = 0)
    //

    /**
     * Returns the edge destination.
     *
     *     @param edge Subdivision edge ID.
     *     @param dstpt Output vertex location.
     *
     *     @return vertex ID.
     */
    public int edgeDst(int edge, Point dstpt) {
        double[] dstpt_out = new double[2];
        int retVal = edgeDst_0(nativeObj, edge, dstpt_out);
        if(dstpt!=null){ dstpt.x = dstpt_out[0]; dstpt.y = dstpt_out[1]; } 
        return retVal;
    }

    /**
     * Returns the edge destination.
     *
     *     @param edge Subdivision edge ID.
     *
     *     @return vertex ID.
     */
    public int edgeDst(int edge) {
        return edgeDst_1(nativeObj, edge);
    }


    //
    // C++:  int cv::Subdiv2D::edgeOrg(int edge, Point2f* orgpt = 0)
    //

    /**
     * Returns the edge origin.
     *
     *     @param edge Subdivision edge ID.
     *     @param orgpt Output vertex location.
     *
     *     @return vertex ID.
     */
    public int edgeOrg(int edge, Point orgpt) {
        double[] orgpt_out = new double[2];
        int retVal = edgeOrg_0(nativeObj, edge, orgpt_out);
        if(orgpt!=null){ orgpt.x = orgpt_out[0]; orgpt.y = orgpt_out[1]; } 
        return retVal;
    }

    /**
     * Returns the edge origin.
     *
     *     @param edge Subdivision edge ID.
     *
     *     @return vertex ID.
     */
    public int edgeOrg(int edge) {
        return edgeOrg_1(nativeObj, edge);
    }


    //
    // C++:  int cv::Subdiv2D::findNearest(Point2f pt, Point2f* nearestPt = 0)
    //

    /**
     * Finds the subdivision vertex closest to the given point.
     *
     *     @param pt Input point.
     *     @param nearestPt Output subdivision vertex point.
     *
     *     The function is another function that locates the input point within the subdivision. It finds the
     *     subdivision vertex that is the closest to the input point. It is not necessarily one of vertices
     *     of the facet containing the input point, though the facet (located using locate() ) is used as a
     *     starting point.
     *
     *     @return vertex ID.
     */
    public int findNearest(Point pt, Point nearestPt) {
        double[] nearestPt_out = new double[2];
        int retVal = findNearest_0(nativeObj, pt.x, pt.y, nearestPt_out);
        if(nearestPt!=null){ nearestPt.x = nearestPt_out[0]; nearestPt.y = nearestPt_out[1]; } 
        return retVal;
    }

    /**
     * Finds the subdivision vertex closest to the given point.
     *
     *     @param pt Input point.
     *
     *     The function is another function that locates the input point within the subdivision. It finds the
     *     subdivision vertex that is the closest to the input point. It is not necessarily one of vertices
     *     of the facet containing the input point, though the facet (located using locate() ) is used as a
     *     starting point.
     *
     *     @return vertex ID.
     */
    public int findNearest(Point pt) {
        return findNearest_1(nativeObj, pt.x, pt.y);
    }


    //
    // C++:  int cv::Subdiv2D::getEdge(int edge, int nextEdgeType)
    //

    /**
     * Returns one of the edges related to the given edge.
     *
     *     @param edge Subdivision edge ID.
     *     @param nextEdgeType Parameter specifying which of the related edges to return.
     *     The following values are possible:
     * <ul>
     *   <li>
     *        NEXT_AROUND_ORG next around the edge origin ( eOnext on the picture below if e is the input edge)
     *   </li>
     *   <li>
     *        NEXT_AROUND_DST next around the edge vertex ( eDnext )
     *   </li>
     *   <li>
     *        PREV_AROUND_ORG previous around the edge origin (reversed eRnext )
     *   </li>
     *   <li>
     *        PREV_AROUND_DST previous around the edge destination (reversed eLnext )
     *   </li>
     *   <li>
     *        NEXT_AROUND_LEFT next around the left facet ( eLnext )
     *   </li>
     *   <li>
     *        NEXT_AROUND_RIGHT next around the right facet ( eRnext )
     *   </li>
     *   <li>
     *        PREV_AROUND_LEFT previous around the left facet (reversed eOnext )
     *   </li>
     *   <li>
     *        PREV_AROUND_RIGHT previous around the right facet (reversed eDnext )
     *   </li>
     * </ul>
     *
     *     ![sample output](pics/quadedge.png)
     *
     *     @return edge ID related to the input edge.
     */
    public int getEdge(int edge, int nextEdgeType) {
        return getEdge_0(nativeObj, edge, nextEdgeType);
    }


    //
    // C++:  int cv::Subdiv2D::insert(Point2f pt)
    //

    /**
     * Insert a single point into a Delaunay triangulation.
     *
     *     @param pt Point to insert.
     *
     *     The function inserts a single point into a subdivision and modifies the subdivision topology
     *     appropriately. If a point with the same coordinates exists already, no new point is added.
     *     @return the ID of the point.
     *
     *     <b>Note:</b> If the point is outside of the triangulation specified rect a runtime error is raised.
     */
    public int insert(Point pt) {
        return insert_0(nativeObj, pt.x, pt.y);
    }


    //
    // C++:  int cv::Subdiv2D::locate(Point2f pt, int& edge, int& vertex)
    //

    /**
     * Returns the location of a point within a Delaunay triangulation.
     *
     *     @param pt Point to locate.
     *     @param edge Output edge that the point belongs to or is located to the right of it.
     *     @param vertex Optional output vertex the input point coincides with.
     *
     *     The function locates the input point within the subdivision and gives one of the triangle edges
     *     or vertices.
     *
     *     @return an integer which specify one of the following five cases for point location:
     * <ul>
     *   <li>
     *       The point falls into some facet. The function returns #PTLOC_INSIDE and edge will contain one of
     *        edges of the facet.
     *   </li>
     *   <li>
     *       The point falls onto the edge. The function returns #PTLOC_ON_EDGE and edge will contain this edge.
     *   </li>
     *   <li>
     *       The point coincides with one of the subdivision vertices. The function returns #PTLOC_VERTEX and
     *        vertex will contain a pointer to the vertex.
     *   </li>
     *   <li>
     *       The point is outside the subdivision reference rectangle. The function returns #PTLOC_OUTSIDE_RECT
     *        and no pointers are filled.
     *   </li>
     *   <li>
     *       One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error
     *        processing mode is selected, #PTLOC_ERROR is returned.
     *   </li>
     * </ul>
     */
    public int locate(Point pt, int[] edge, int[] vertex) {
        double[] edge_out = new double[1];
        double[] vertex_out = new double[1];
        int retVal = locate_0(nativeObj, pt.x, pt.y, edge_out, vertex_out);
        if(edge!=null) edge[0] = (int)edge_out[0];
        if(vertex!=null) vertex[0] = (int)vertex_out[0];
        return retVal;
    }


    //
    // C++:  int cv::Subdiv2D::nextEdge(int edge)
    //

    /**
     * Returns next edge around the edge origin.
     *
     *     @param edge Subdivision edge ID.
     *
     *     @return an integer which is next edge ID around the edge origin: eOnext on the
     *     picture above if e is the input edge).
     */
    public int nextEdge(int edge) {
        return nextEdge_0(nativeObj, edge);
    }


    //
    // C++:  int cv::Subdiv2D::rotateEdge(int edge, int rotate)
    //

    /**
     * Returns another edge of the same quad-edge.
     *
     *     @param edge Subdivision edge ID.
     *     @param rotate Parameter specifying which of the edges of the same quad-edge as the input
     *     one to return. The following values are possible:
     * <ul>
     *   <li>
     *        0 - the input edge ( e on the picture below if e is the input edge)
     *   </li>
     *   <li>
     *        1 - the rotated edge ( eRot )
     *   </li>
     *   <li>
     *        2 - the reversed edge (reversed e (in green))
     *   </li>
     *   <li>
     *        3 - the reversed rotated edge (reversed eRot (in green))
     *   </li>
     * </ul>
     *
     *     @return one of the edges ID of the same quad-edge as the input edge.
     */
    public int rotateEdge(int edge, int rotate) {
        return rotateEdge_0(nativeObj, edge, rotate);
    }


    //
    // C++:  int cv::Subdiv2D::symEdge(int edge)
    //

    public int symEdge(int edge) {
        return symEdge_0(nativeObj, edge);
    }


    //
    // C++:  void cv::Subdiv2D::getEdgeList(vector_Vec4f& edgeList)
    //

    /**
     * Returns a list of all edges.
     *
     *     @param edgeList Output vector.
     *
     *     The function gives each edge as a 4 numbers vector, where each two are one of the edge
     *     vertices. i.e. org_x = v[0], org_y = v[1], dst_x = v[2], dst_y = v[3].
     */
    public void getEdgeList(MatOfFloat4 edgeList) {
        Mat edgeList_mat = edgeList;
        getEdgeList_0(nativeObj, edgeList_mat.nativeObj);
    }


    //
    // C++:  void cv::Subdiv2D::getLeadingEdgeList(vector_int& leadingEdgeList)
    //

    /**
     * Returns a list of the leading edge ID connected to each triangle.
     *
     *     @param leadingEdgeList Output vector.
     *
     *     The function gives one edge ID for each triangle.
     */
    public void getLeadingEdgeList(MatOfInt leadingEdgeList) {
        Mat leadingEdgeList_mat = leadingEdgeList;
        getLeadingEdgeList_0(nativeObj, leadingEdgeList_mat.nativeObj);
    }


    //
    // C++:  void cv::Subdiv2D::getTriangleList(vector_Vec6f& triangleList)
    //

    /**
     * Returns a list of all triangles.
     *
     *     @param triangleList Output vector.
     *
     *     The function gives each triangle as a 6 numbers vector, where each two are one of the triangle
     *     vertices. i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5].
     */
    public void getTriangleList(MatOfFloat6 triangleList) {
        Mat triangleList_mat = triangleList;
        getTriangleList_0(nativeObj, triangleList_mat.nativeObj);
    }


    //
    // C++:  void cv::Subdiv2D::getVoronoiFacetList(vector_int idx, vector_vector_Point2f& facetList, vector_Point2f& facetCenters)
    //

    /**
     * Returns a list of all Voronoi facets.
     *
     *     @param idx Vector of vertices IDs to consider. For all vertices you can pass empty vector.
     *     @param facetList Output vector of the Voronoi facets.
     *     @param facetCenters Output vector of the Voronoi facets center points.
     */
    public void getVoronoiFacetList(MatOfInt idx, List<MatOfPoint2f> facetList, MatOfPoint2f facetCenters) {
        Mat idx_mat = idx;
        Mat facetList_mat = new Mat();
        Mat facetCenters_mat = facetCenters;
        getVoronoiFacetList_0(nativeObj, idx_mat.nativeObj, facetList_mat.nativeObj, facetCenters_mat.nativeObj);
        Converters.Mat_to_vector_vector_Point2f(facetList_mat, facetList);
        facetList_mat.release();
    }


    //
    // C++:  void cv::Subdiv2D::initDelaunay(Rect rect)
    //

    /**
     * Creates a new empty Delaunay subdivision
     *
     *     @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
     */
    public void initDelaunay(Rect rect) {
        initDelaunay_0(nativeObj, rect.x, rect.y, rect.width, rect.height);
    }


    //
    // C++:  void cv::Subdiv2D::insert(vector_Point2f ptvec)
    //

    /**
     * Insert multiple points into a Delaunay triangulation.
     *
     *     @param ptvec Points to insert.
     *
     *     The function inserts a vector of points into a subdivision and modifies the subdivision topology
     *     appropriately.
     */
    public void insert(MatOfPoint2f ptvec) {
        Mat ptvec_mat = ptvec;
        insert_1(nativeObj, ptvec_mat.nativeObj);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::Subdiv2D::Subdiv2D(Rect rect)
    private static native long Subdiv2D_0(int rect_x, int rect_y, int rect_width, int rect_height);

    // C++:   cv::Subdiv2D::Subdiv2D()
    private static native long Subdiv2D_1();

    // C++:  Point2f cv::Subdiv2D::getVertex(int vertex, int* firstEdge = 0)
    private static native double[] getVertex_0(long nativeObj, int vertex, double[] firstEdge_out);
    private static native double[] getVertex_1(long nativeObj, int vertex);

    // C++:  int cv::Subdiv2D::edgeDst(int edge, Point2f* dstpt = 0)
    private static native int edgeDst_0(long nativeObj, int edge, double[] dstpt_out);
    private static native int edgeDst_1(long nativeObj, int edge);

    // C++:  int cv::Subdiv2D::edgeOrg(int edge, Point2f* orgpt = 0)
    private static native int edgeOrg_0(long nativeObj, int edge, double[] orgpt_out);
    private static native int edgeOrg_1(long nativeObj, int edge);

    // C++:  int cv::Subdiv2D::findNearest(Point2f pt, Point2f* nearestPt = 0)
    private static native int findNearest_0(long nativeObj, double pt_x, double pt_y, double[] nearestPt_out);
    private static native int findNearest_1(long nativeObj, double pt_x, double pt_y);

    // C++:  int cv::Subdiv2D::getEdge(int edge, int nextEdgeType)
    private static native int getEdge_0(long nativeObj, int edge, int nextEdgeType);

    // C++:  int cv::Subdiv2D::insert(Point2f pt)
    private static native int insert_0(long nativeObj, double pt_x, double pt_y);

    // C++:  int cv::Subdiv2D::locate(Point2f pt, int& edge, int& vertex)
    private static native int locate_0(long nativeObj, double pt_x, double pt_y, double[] edge_out, double[] vertex_out);

    // C++:  int cv::Subdiv2D::nextEdge(int edge)
    private static native int nextEdge_0(long nativeObj, int edge);

    // C++:  int cv::Subdiv2D::rotateEdge(int edge, int rotate)
    private static native int rotateEdge_0(long nativeObj, int edge, int rotate);

    // C++:  int cv::Subdiv2D::symEdge(int edge)
    private static native int symEdge_0(long nativeObj, int edge);

    // C++:  void cv::Subdiv2D::getEdgeList(vector_Vec4f& edgeList)
    private static native void getEdgeList_0(long nativeObj, long edgeList_mat_nativeObj);

    // C++:  void cv::Subdiv2D::getLeadingEdgeList(vector_int& leadingEdgeList)
    private static native void getLeadingEdgeList_0(long nativeObj, long leadingEdgeList_mat_nativeObj);

    // C++:  void cv::Subdiv2D::getTriangleList(vector_Vec6f& triangleList)
    private static native void getTriangleList_0(long nativeObj, long triangleList_mat_nativeObj);

    // C++:  void cv::Subdiv2D::getVoronoiFacetList(vector_int idx, vector_vector_Point2f& facetList, vector_Point2f& facetCenters)
    private static native void getVoronoiFacetList_0(long nativeObj, long idx_mat_nativeObj, long facetList_mat_nativeObj, long facetCenters_mat_nativeObj);

    // C++:  void cv::Subdiv2D::initDelaunay(Rect rect)
    private static native void initDelaunay_0(long nativeObj, int rect_x, int rect_y, int rect_width, int rect_height);

    // C++:  void cv::Subdiv2D::insert(vector_Point2f ptvec)
    private static native void insert_1(long nativeObj, long ptvec_mat_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
