/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2013, Open Source Robotics Foundation, Inc.
 * Copyright (c) 2010-2012, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *  * Neither the name of Open Source Robotics Foundation, Inc. nor
 *    the names of its contributors may be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PCL_CONVERSIONS_H__
#define PCL_CONVERSIONS_H__

#include <cstddef>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <message_filters/message_event.h>
#include <message_filters/message_traits.h>

#include <pcl/conversions.h>

#include <pcl/PCLHeader.h>
#include <std_msgs/msg/header.hpp>

#include <pcl/PCLImage.h>
#include <sensor_msgs/msg/image.hpp>

#include <pcl/PCLPointField.h>
#include <sensor_msgs/msg/point_field.hpp>

#include <pcl/PCLPointCloud2.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/PointIndices.h>
#include <pcl_msgs/msg/point_indices.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl_msgs/msg/model_coefficients.hpp>

#include <pcl/Vertices.h>
#include <pcl_msgs/msg/vertices.hpp>

#include <pcl/PolygonMesh.h>
#include <pcl_msgs/msg/polygon_mesh.hpp>

#include <pcl/io/pcd_io.h>

#include <Eigen/StdVector>
#include <Eigen/Geometry>

namespace pcl_conversions {

  /** PCLHeader <=> Header **/

  inline
  void fromPCL(const std::uint64_t &pcl_stamp, rclcpp::Time &stamp)
  {
    stamp = rclcpp::Time(
      static_cast<rcl_time_point_value_t>(pcl_stamp * 1000ull));  // Convert from us to ns
  }

  inline
  void toPCL(const rclcpp::Time &stamp, std::uint64_t &pcl_stamp)
  {
    pcl_stamp = static_cast<std::uint64_t>(stamp.nanoseconds()) / 1000ull;  // Convert from ns to us
  }

  inline
  rclcpp::Time fromPCL(const std::uint64_t &pcl_stamp)
  {
    rclcpp::Time stamp;
    fromPCL(pcl_stamp, stamp);
    return stamp;
  }

  inline
  std::uint64_t toPCL(const rclcpp::Time &stamp)
  {
    std::uint64_t pcl_stamp;
    toPCL(stamp, pcl_stamp);
    return pcl_stamp;
  }

  /** PCLHeader <=> Header **/

  inline
  void fromPCL(const pcl::PCLHeader &pcl_header, std_msgs::msg::Header &header)
  {
    header.stamp = fromPCL(pcl_header.stamp);
    header.frame_id = pcl_header.frame_id;
  }

  inline
  void toPCL(const std_msgs::msg::Header &header, pcl::PCLHeader &pcl_header)
  {
    toPCL(header.stamp, pcl_header.stamp);
    // TODO(clalancette): Seq doesn't exist in the ROS2 header
    // anymore.  wjwwood suggests that we might be able to get this
    // information from the middleware in the future, but for now we
    // just set it to 0.
    pcl_header.seq = 0;
    pcl_header.frame_id = header.frame_id;
  }

  inline
  std_msgs::msg::Header fromPCL(const pcl::PCLHeader &pcl_header)
  {
    std_msgs::msg::Header header;
    fromPCL(pcl_header, header);
    return header;
  }

  inline
  pcl::PCLHeader toPCL(const std_msgs::msg::Header &header)
  {
    pcl::PCLHeader pcl_header;
    toPCL(header, pcl_header);
    return pcl_header;
  }

  /** PCLImage <=> Image **/

  inline
  void copyPCLImageMetaData(const pcl::PCLImage &pcl_image, sensor_msgs::msg::Image &image)
  {
    fromPCL(pcl_image.header, image.header);
    image.height = pcl_image.height;
    image.width = pcl_image.width;
    image.encoding = pcl_image.encoding;
    image.is_bigendian = pcl_image.is_bigendian;
    image.step = pcl_image.step;
  }

  inline
  void fromPCL(const pcl::PCLImage &pcl_image, sensor_msgs::msg::Image &image)
  {
    copyPCLImageMetaData(pcl_image, image);
    image.data = pcl_image.data;
  }

  inline
  void moveFromPCL(pcl::PCLImage &pcl_image, sensor_msgs::msg::Image &image)
  {
    copyPCLImageMetaData(pcl_image, image);
    image.data.swap(pcl_image.data);
  }

  inline
  void copyImageMetaData(const sensor_msgs::msg::Image &image, pcl::PCLImage &pcl_image)
  {
    toPCL(image.header, pcl_image.header);
    pcl_image.height = image.height;
    pcl_image.width = image.width;
    pcl_image.encoding = image.encoding;
    pcl_image.is_bigendian = image.is_bigendian;
    pcl_image.step = image.step;
  }

  inline
  void toPCL(const sensor_msgs::msg::Image &image, pcl::PCLImage &pcl_image)
  {
    copyImageMetaData(image, pcl_image);
    pcl_image.data = image.data;
  }

  inline
  void moveToPCL(sensor_msgs::msg::Image &image, pcl::PCLImage &pcl_image)
  {
    copyImageMetaData(image, pcl_image);
    pcl_image.data.swap(image.data);
  }

  /** PCLPointField <=> PointField **/

  inline
  void fromPCL(const pcl::PCLPointField &pcl_pf, sensor_msgs::msg::PointField &pf)
  {
    pf.name = pcl_pf.name;
    pf.offset = pcl_pf.offset;
    pf.datatype = pcl_pf.datatype;
    pf.count = pcl_pf.count;
  }

  inline
  void fromPCL(const std::vector<pcl::PCLPointField> &pcl_pfs, std::vector<sensor_msgs::msg::PointField> &pfs)
  {
    pfs.resize(pcl_pfs.size());
    std::vector<pcl::PCLPointField>::const_iterator it = pcl_pfs.begin();
    size_t i = 0;
    for(; it != pcl_pfs.end(); ++it, ++i) {
      fromPCL(*(it), pfs[i]);
    }
    std::sort(pfs.begin(), pfs.end(), [](const auto& field_a, const auto& field_b)
                                      {
                                        return field_a.offset < field_b.offset;
                                      });
  }

  inline
  void toPCL(const sensor_msgs::msg::PointField &pf, pcl::PCLPointField &pcl_pf)
  {
    pcl_pf.name = pf.name;
    pcl_pf.offset = pf.offset;
    pcl_pf.datatype = pf.datatype;
    pcl_pf.count = pf.count;
  }

  inline
  void toPCL(const std::vector<sensor_msgs::msg::PointField> &pfs, std::vector<pcl::PCLPointField> &pcl_pfs)
  {
    pcl_pfs.resize(pfs.size());
    std::vector<sensor_msgs::msg::PointField>::const_iterator it = pfs.begin();
    size_t i = 0;
    for(; it != pfs.end(); ++it, ++i) {
      toPCL(*(it), pcl_pfs[i]);
    }
  }

  /** PCLPointCloud2 <=> PointCloud2 **/

  inline
  void copyPCLPointCloud2MetaData(const pcl::PCLPointCloud2 &pcl_pc2, sensor_msgs::msg::PointCloud2 &pc2)
  {
    fromPCL(pcl_pc2.header, pc2.header);
    pc2.height = pcl_pc2.height;
    pc2.width = pcl_pc2.width;
    fromPCL(pcl_pc2.fields, pc2.fields);
    pc2.is_bigendian = pcl_pc2.is_bigendian;
    pc2.point_step = pcl_pc2.point_step;
    pc2.row_step = pcl_pc2.row_step;
    pc2.is_dense = pcl_pc2.is_dense;
  }

  inline
  void fromPCL(const pcl::PCLPointCloud2 &pcl_pc2, sensor_msgs::msg::PointCloud2 &pc2)
  {
    copyPCLPointCloud2MetaData(pcl_pc2, pc2);
    pc2.data = pcl_pc2.data;
  }

  inline
  void moveFromPCL(pcl::PCLPointCloud2 &pcl_pc2, sensor_msgs::msg::PointCloud2 &pc2)
  {
    copyPCLPointCloud2MetaData(pcl_pc2, pc2);
    pc2.data.swap(pcl_pc2.data);
  }

  inline
  void copyPointCloud2MetaData(const sensor_msgs::msg::PointCloud2 &pc2, pcl::PCLPointCloud2 &pcl_pc2)
  {
    toPCL(pc2.header, pcl_pc2.header);
    pcl_pc2.height = pc2.height;
    pcl_pc2.width = pc2.width;
    toPCL(pc2.fields, pcl_pc2.fields);
    pcl_pc2.is_bigendian = pc2.is_bigendian;
    pcl_pc2.point_step = pc2.point_step;
    pcl_pc2.row_step = pc2.row_step;
    pcl_pc2.is_dense = pc2.is_dense;
  }

  inline
  void toPCL(const sensor_msgs::msg::PointCloud2 &pc2, pcl::PCLPointCloud2 &pcl_pc2)
  {
    copyPointCloud2MetaData(pc2, pcl_pc2);
    pcl_pc2.data = pc2.data;
  }

  inline
  void moveToPCL(sensor_msgs::msg::PointCloud2 &pc2, pcl::PCLPointCloud2 &pcl_pc2)
  {
    copyPointCloud2MetaData(pc2, pcl_pc2);
    pcl_pc2.data.swap(pc2.data);
  }

  /** pcl::PointIndices <=> pcl_msgs::PointIndices **/

  inline
  void fromPCL(const pcl::PointIndices &pcl_pi, pcl_msgs::msg::PointIndices &pi)
  {
    fromPCL(pcl_pi.header, pi.header);
    pi.indices = pcl_pi.indices;
  }

  inline
  void moveFromPCL(pcl::PointIndices &pcl_pi, pcl_msgs::msg::PointIndices &pi)
  {
    fromPCL(pcl_pi.header, pi.header);
    pi.indices.swap(pcl_pi.indices);
  }

  inline
  void toPCL(const pcl_msgs::msg::PointIndices &pi, pcl::PointIndices &pcl_pi)
  {
    toPCL(pi.header, pcl_pi.header);
    pcl_pi.indices = pi.indices;
  }

  inline
  void moveToPCL(pcl_msgs::msg::PointIndices &pi, pcl::PointIndices &pcl_pi)
  {
    toPCL(pi.header, pcl_pi.header);
    pcl_pi.indices.swap(pi.indices);
  }

  /** pcl::ModelCoefficients <=> pcl_msgs::ModelCoefficients **/

  inline
  void fromPCL(const pcl::ModelCoefficients &pcl_mc, pcl_msgs::msg::ModelCoefficients &mc)
  {
    fromPCL(pcl_mc.header, mc.header);
    mc.values = pcl_mc.values;
  }

  inline
  void moveFromPCL(pcl::ModelCoefficients &pcl_mc, pcl_msgs::msg::ModelCoefficients &mc)
  {
    fromPCL(pcl_mc.header, mc.header);
    mc.values.swap(pcl_mc.values);
  }

  inline
  void toPCL(const pcl_msgs::msg::ModelCoefficients &mc, pcl::ModelCoefficients &pcl_mc)
  {
    toPCL(mc.header, pcl_mc.header);
    pcl_mc.values = mc.values;
  }

  inline
  void moveToPCL(pcl_msgs::msg::ModelCoefficients &mc, pcl::ModelCoefficients &pcl_mc)
  {
    toPCL(mc.header, pcl_mc.header);
    pcl_mc.values.swap(mc.values);
  }

  /** pcl::Vertices <=> pcl_msgs::Vertices **/

  namespace internal
  {
    template <class T>
    inline void move(std::vector<T> &a, std::vector<T> &b)
    {
      b.swap(a);
    }

    template <class T1, class T2>
    inline void move(std::vector<T1> &a, std::vector<T2> &b)
    {
      b.assign(a.cbegin(), a.cend());
    }
  }

  inline
  void fromPCL(const pcl::Vertices &pcl_vert, pcl_msgs::msg::Vertices &vert)
  {
    vert.vertices.assign(pcl_vert.vertices.cbegin(), pcl_vert.vertices.cend());
  }

  inline
  void fromPCL(const std::vector<pcl::Vertices> &pcl_verts, std::vector<pcl_msgs::msg::Vertices> &verts)
  {
    verts.resize(pcl_verts.size());
    std::vector<pcl::Vertices>::const_iterator it = pcl_verts.begin();
    std::vector<pcl_msgs::msg::Vertices>::iterator jt = verts.begin();
    for (; it != pcl_verts.end() && jt != verts.end(); ++it, ++jt) {
      fromPCL(*(it), *(jt));
    }
  }

  inline
  void moveFromPCL(pcl::Vertices &pcl_vert, pcl_msgs::msg::Vertices &vert)
  {
    internal::move(pcl_vert.vertices, vert.vertices);
  }

  inline
  void fromPCL(std::vector<pcl::Vertices> &pcl_verts, std::vector<pcl_msgs::msg::Vertices> &verts)
  {
    verts.resize(pcl_verts.size());
    std::vector<pcl::Vertices>::iterator it = pcl_verts.begin();
    std::vector<pcl_msgs::msg::Vertices>::iterator jt = verts.begin();
    for (; it != pcl_verts.end() && jt != verts.end(); ++it, ++jt) {
      moveFromPCL(*(it), *(jt));
    }
  }

  inline
  void toPCL(const pcl_msgs::msg::Vertices &vert, pcl::Vertices &pcl_vert)
  {
    pcl_vert.vertices.assign(vert.vertices.cbegin(), vert.vertices.cend());
  }

  inline
  void toPCL(const std::vector<pcl_msgs::msg::Vertices> &verts, std::vector<pcl::Vertices> &pcl_verts)
  {
    pcl_verts.resize(verts.size());
    std::vector<pcl_msgs::msg::Vertices>::const_iterator it = verts.begin();
    std::vector<pcl::Vertices>::iterator jt = pcl_verts.begin();
    for (; it != verts.end() && jt != pcl_verts.end(); ++it, ++jt) {
      toPCL(*(it), *(jt));
    }
  }

  inline
  void moveToPCL(pcl_msgs::msg::Vertices &vert, pcl::Vertices &pcl_vert)
  {
    internal::move(vert.vertices, pcl_vert.vertices);
  }

  inline
  void moveToPCL(std::vector<pcl_msgs::msg::Vertices> &verts, std::vector<pcl::Vertices> &pcl_verts)
  {
    pcl_verts.resize(verts.size());
    std::vector<pcl_msgs::msg::Vertices>::iterator it = verts.begin();
    std::vector<pcl::Vertices>::iterator jt = pcl_verts.begin();
    for (; it != verts.end() && jt != pcl_verts.end(); ++it, ++jt) {
      moveToPCL(*(it), *(jt));
    }
  }

  /** pcl::PolygonMesh <=> pcl_msgs::PolygonMesh **/

  inline
  void fromPCL(const pcl::PolygonMesh &pcl_mesh, pcl_msgs::msg::PolygonMesh &mesh)
  {
    fromPCL(pcl_mesh.header, mesh.header);
    fromPCL(pcl_mesh.cloud, mesh.cloud);
    fromPCL(pcl_mesh.polygons, mesh.polygons);
  }

  inline
  void moveFromPCL(pcl::PolygonMesh &pcl_mesh, pcl_msgs::msg::PolygonMesh &mesh)
  {
    fromPCL(pcl_mesh.header, mesh.header);
    moveFromPCL(pcl_mesh.cloud, mesh.cloud);
  }

  inline
  void toPCL(const pcl_msgs::msg::PolygonMesh &mesh, pcl::PolygonMesh &pcl_mesh)
  {
    toPCL(mesh.header, pcl_mesh.header);
    toPCL(mesh.cloud, pcl_mesh.cloud);
    toPCL(mesh.polygons, pcl_mesh.polygons);
  }

  inline
  void moveToPCL(pcl_msgs::msg::PolygonMesh &mesh, pcl::PolygonMesh &pcl_mesh)
  {
    toPCL(mesh.header, pcl_mesh.header);
    moveToPCL(mesh.cloud, pcl_mesh.cloud);
    moveToPCL(mesh.polygons, pcl_mesh.polygons);
  }

} // namespace pcl_conversions

namespace pcl {

  /** Overload pcl::getFieldIndex **/

  inline int getFieldIndex(const sensor_msgs::msg::PointCloud2 &cloud, const std::string &field_name)
  {
    // Get the index we need
    for (size_t d = 0; d < cloud.fields.size(); ++d) {
      if (cloud.fields[d].name == field_name) {
        return (static_cast<int>(d));
      }
    }
    return (-1);
  }

  /** Overload pcl::getFieldsList **/

  inline std::string getFieldsList(const sensor_msgs::msg::PointCloud2 &cloud)
  {
    std::string result;
    for (size_t i = 0; i < cloud.fields.size () - 1; ++i) {
      result += cloud.fields[i].name + " ";
    }
    result += cloud.fields[cloud.fields.size () - 1].name;
    return (result);
  }

  /** Provide pcl::toROSMsg **/

  inline
  void toROSMsg(const sensor_msgs::msg::PointCloud2 &cloud, sensor_msgs::msg::Image &image)
  {
    pcl::PCLPointCloud2 pcl_cloud;
    pcl_conversions::toPCL(cloud, pcl_cloud);
    pcl::PCLImage pcl_image;
    pcl::toPCLPointCloud2(pcl_cloud, pcl_image);
    pcl_conversions::moveFromPCL(pcl_image, image);
  }

  inline
  void moveToROSMsg(sensor_msgs::msg::PointCloud2 &cloud, sensor_msgs::msg::Image &image)
  {
    pcl::PCLPointCloud2 pcl_cloud;
    pcl_conversions::moveToPCL(cloud, pcl_cloud);
    pcl::PCLImage pcl_image;
    pcl::toPCLPointCloud2(pcl_cloud, pcl_image);
    pcl_conversions::moveFromPCL(pcl_image, image);
  }

  template<typename T> void
  toROSMsg (const pcl::PointCloud<T> &cloud, sensor_msgs::msg::Image& msg)
  {
    // Ease the user's burden on specifying width/height for unorganized datasets
    if (cloud.width == 0 && cloud.height == 0)
    {
      throw std::runtime_error("Needs to be a dense like cloud!!");
    }
    else
    {
      if (cloud.points.size () != cloud.width * cloud.height)
        throw std::runtime_error("The width and height do not match the cloud size!");
      msg.height = cloud.height;
      msg.width = cloud.width;
    }

    // sensor_msgs::image_encodings::BGR8;
    msg.encoding = "bgr8";
    msg.step = msg.width * sizeof (std::uint8_t) * 3;
    msg.data.resize (msg.step * msg.height);
    for (size_t y = 0; y < cloud.height; y++)
    {
      for (size_t x = 0; x < cloud.width; x++)
      {
        std::uint8_t * pixel = &(msg.data[y * msg.step + x * 3]);
        memcpy (pixel, &cloud (x, y).rgb, 3 * sizeof(std::uint8_t));
      }
    }
  }

  /** Provide to/fromROSMsg for sensor_msgs::msg::PointCloud2 <=> pcl::PointCloud<T> **/

  template<typename T>
  void toROSMsg(const pcl::PointCloud<T> &pcl_cloud, sensor_msgs::msg::PointCloud2 &cloud)
  {
    pcl::PCLPointCloud2 pcl_pc2;
#if PCL_VERSION_COMPARE(>=, 1, 14, 1)
    // if PCL version is recent enough, request that all padding be removed to make the msg as small as possible
    pcl::toPCLPointCloud2(pcl_cloud, pcl_pc2, false);
#else
    pcl::toPCLPointCloud2(pcl_cloud, pcl_pc2);
#endif
    pcl_conversions::moveFromPCL(pcl_pc2, cloud);
  }

  template<typename T>
  void fromROSMsg(const sensor_msgs::msg::PointCloud2 &cloud, pcl::PointCloud<T> &pcl_cloud)
  {
    pcl::PCLPointCloud2 pcl_pc2;
#if PCL_VERSION_COMPARE(>=, 1, 13, 1)
    pcl_conversions::copyPointCloud2MetaData(cloud, pcl_pc2); // Like pcl_conversions::toPCL, but does not copy the binary data
    pcl::MsgFieldMap field_map;
    pcl::createMapping<T> (pcl_pc2.fields, field_map);
    pcl::fromPCLPointCloud2(pcl_pc2, pcl_cloud, field_map, &cloud.data[0]);
#else
    pcl_conversions::toPCL(cloud, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, pcl_cloud);
#endif
  }

  template<typename T>
  void moveFromROSMsg(sensor_msgs::msg::PointCloud2 &cloud, pcl::PointCloud<T> &pcl_cloud)
  {
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::moveToPCL(cloud, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, pcl_cloud);
  }

  /** Overload pcl::createMapping **/

  template<typename PointT>
  void createMapping(const std::vector<sensor_msgs::msg::PointField>& msg_fields, MsgFieldMap& field_map)
  {
    std::vector<pcl::PCLPointField> pcl_msg_fields;
    pcl_conversions::toPCL(msg_fields, pcl_msg_fields);
    return createMapping<PointT>(pcl_msg_fields, field_map);
  }

  namespace io {

    /** Overload pcl::io::savePCDFile **/

    inline int
    savePCDFile(const std::string &file_name, const sensor_msgs::msg::PointCloud2 &cloud,
                const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (),
                const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
                const bool binary_mode = false)
    {
      pcl::PCLPointCloud2 pcl_cloud;
      pcl_conversions::toPCL(cloud, pcl_cloud);
      return pcl::io::savePCDFile(file_name, pcl_cloud, origin, orientation, binary_mode);
    }

    inline int
    destructiveSavePCDFile(const std::string &file_name, sensor_msgs::msg::PointCloud2 &cloud,
                           const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (),
                           const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
                           const bool binary_mode = false)
    {
      pcl::PCLPointCloud2 pcl_cloud;
      pcl_conversions::moveToPCL(cloud, pcl_cloud);
      return pcl::io::savePCDFile(file_name, pcl_cloud, origin, orientation, binary_mode);
    }

    /** Overload pcl::io::loadPCDFile **/

    inline int loadPCDFile(const std::string &file_name, sensor_msgs::msg::PointCloud2 &cloud)
    {
      pcl::PCLPointCloud2 pcl_cloud;
      int ret = pcl::io::loadPCDFile(file_name, pcl_cloud);
      pcl_conversions::moveFromPCL(pcl_cloud, cloud);
      return ret;
    }

  } // namespace io

  /** Overload asdf **/

  inline
  bool concatenatePointCloud (const sensor_msgs::msg::PointCloud2 &cloud1,
                              const sensor_msgs::msg::PointCloud2 &cloud2,
                              sensor_msgs::msg::PointCloud2 &cloud_out)
  {
    //if one input cloud has no points, but the other input does, just return the cloud with points
    if (cloud1.width * cloud1.height == 0 && cloud2.width * cloud2.height > 0)
    {
      cloud_out = cloud2;
      return (true);
    }
    else if (cloud1.width*cloud1.height > 0 && cloud2.width*cloud2.height == 0)
    {
      cloud_out = cloud1;
      return (true);
    }

    bool strip = false;
    for (size_t i = 0; i < cloud1.fields.size (); ++i)
      if (cloud1.fields[i].name == "_")
        strip = true;

    for (size_t i = 0; i < cloud2.fields.size (); ++i)
      if (cloud2.fields[i].name == "_")
        strip = true;

    if (!strip && cloud1.fields.size () != cloud2.fields.size ())
    {
      PCL_ERROR ("[pcl::concatenatePointCloud] Number of fields in cloud1 (%u) != Number of fields in cloud2 (%u)\n", cloud1.fields.size (), cloud2.fields.size ());
      return (false);
    }

    // Copy cloud1 into cloud_out
    cloud_out = cloud1;
    size_t nrpts = cloud_out.data.size ();
    // Height = 1 => no more organized
    cloud_out.width    = cloud1.width * cloud1.height + cloud2.width * cloud2.height;
    cloud_out.height   = 1;
    cloud_out.row_step = cloud_out.width * cloud_out.point_step;
    if (!cloud1.is_dense || !cloud2.is_dense)
      cloud_out.is_dense = false;
    else
      cloud_out.is_dense = true;

    // We need to strip the extra padding fields
    if (strip)
    {
      // Get the field sizes for the second cloud
      std::vector<sensor_msgs::msg::PointField> fields2;
      std::vector<size_t> fields2_sizes;
      for (size_t j = 0; j < cloud2.fields.size (); ++j)
      {
        if (cloud2.fields[j].name == "_")
          continue;

        fields2_sizes.push_back(
          cloud2.fields[j].count *
          static_cast<size_t>(pcl::getFieldSize(cloud2.fields[j].datatype)));
        fields2.push_back(cloud2.fields[j]);
      }

      cloud_out.data.resize (nrpts + (cloud2.width * cloud2.height) * cloud_out.point_step);

      // Copy the second cloud
      for (size_t cp = 0; cp < cloud2.width * cloud2.height; ++cp)
      {
        size_t i = 0;
        for (size_t j = 0; j < fields2.size (); ++j)
        {
          if (cloud1.fields[i].name == "_")
          {
            ++i;
            continue;
          }

          // We're fine with the special RGB vs RGBA use case
          if ((cloud1.fields[i].name == "rgb" && fields2[j].name == "rgba") ||
              (cloud1.fields[i].name == "rgba" && fields2[j].name == "rgb") ||
              (cloud1.fields[i].name == fields2[j].name))
          {
            memcpy (reinterpret_cast<char*> (&cloud_out.data[nrpts + cp * cloud1.point_step + cloud1.fields[i].offset]),
                    reinterpret_cast<const char*> (&cloud2.data[cp * cloud2.point_step + cloud2.fields[j].offset]),
                    fields2_sizes[j]);
            ++i;  // increment the field size i
          }
        }
      }
    }
    else
    {
      for (size_t i = 0; i < cloud1.fields.size (); ++i)
      {
        // We're fine with the special RGB vs RGBA use case
        if ((cloud1.fields[i].name == "rgb" && cloud2.fields[i].name == "rgba") ||
            (cloud1.fields[i].name == "rgba" && cloud2.fields[i].name == "rgb"))
          continue;
        // Otherwise we need to make sure the names are the same
        if (cloud1.fields[i].name != cloud2.fields[i].name)
        {
          PCL_ERROR ("[pcl::concatenatePointCloud] Name of field %d in cloud1, %s, does not match name in cloud2, %s\n", i, cloud1.fields[i].name.c_str (), cloud2.fields[i].name.c_str ());
          return (false);
        }
      }
      cloud_out.data.resize (nrpts + cloud2.data.size ());
      memcpy (&cloud_out.data[nrpts], &cloud2.data[0], cloud2.data.size ());
    }
    return (true);
  }

} // namespace pcl

/* TODO when ROS2 type masquerading is implemented */ 
/**
namespace ros
{
  template<>
  struct DefaultMessageCreator<pcl::PCLPointCloud2>
  {
    std::shared_ptr<pcl::PCLPointCloud2> operator() ()
    {
      std::shared_ptr<pcl::PCLPointCloud2> msg(new pcl::PCLPointCloud2());
      return msg;
    }
  };
  
  namespace message_traits
  {
    template<>
    struct MD5Sum<pcl::PCLPointCloud2>
    {
      static const char* value() { return MD5Sum<sensor_msgs::msg::PointCloud2>::value(); }
      static const char* value(const pcl::PCLPointCloud2&) { return value(); }

      static const uint64_t static_value1 = MD5Sum<sensor_msgs::msg::PointCloud2>::static_value1;
      static const uint64_t static_value2 = MD5Sum<sensor_msgs::msg::PointCloud2>::static_value2;

      // If the definition of sensor_msgs/PointCloud2 changes, we'll get a compile error here.
      static_assert(static_value1 == 0x1158d486dd51d683ULL);
      static_assert(static_value2 == 0xce2f1be655c3c181ULL);
    };

    template<>
    struct DataType<pcl::PCLPointCloud2>
    {
      static const char* value() { return DataType<sensor_msgs::msg::PointCloud2>::value(); }
      static const char* value(const pcl::PCLPointCloud2&) { return value(); }
    };

    template<>
    struct Definition<pcl::PCLPointCloud2>
    {
      static const char* value() { return Definition<sensor_msgs::msg::PointCloud2>::value(); }
      static const char* value(const pcl::PCLPointCloud2&) { return value(); }
    };

    template<> struct HasHeader<pcl::PCLPointCloud2> : std::true_type {};
  } // namespace message_filters::message_traits

  namespace serialization
  {
  **/
    /*
     * Provide a custom serialization for pcl::PCLPointCloud2
     */
    /**
    template<>
    struct Serializer<pcl::PCLPointCloud2>
    {
      template<typename Stream>
      inline static void write(Stream& stream, const pcl::PCLPointCloud2& m)
      {
        std_msgs::msg::Header header;
        pcl_conversions::fromPCL(m.header, header);
        stream.next(header);
        stream.next(m.height);
        stream.next(m.width);
        std::vector<sensor_msgs::msg::PointField> pfs;
        pcl_conversions::fromPCL(m.fields, pfs);
        stream.next(pfs);
        stream.next(m.is_bigendian);
        stream.next(m.point_step);
        stream.next(m.row_step);
        stream.next(m.data);
        stream.next(m.is_dense);
      }

      template<typename Stream>
      inline static void read(Stream& stream, pcl::PCLPointCloud2& m)
      {
        std_msgs::msg::Header header;
        stream.next(header);
        pcl_conversions::toPCL(header, m.header);
        stream.next(m.height);
        stream.next(m.width);
        std::vector<sensor_msgs::msg::PointField> pfs;
        stream.next(pfs);
        pcl_conversions::toPCL(pfs, m.fields);
        stream.next(m.is_bigendian);
        stream.next(m.point_step);
        stream.next(m.row_step);
        stream.next(m.data);
        stream.next(m.is_dense);
      }

      inline static uint32_t serializedLength(const pcl::PCLPointCloud2& m)
      {
        uint32_t length = 0;

        std_msgs::msg::Header header;
        pcl_conversions::fromPCL(m.header, header);
        length += serializationLength(header);
        length += 4; // height
        length += 4; // width
        std::vector<sensor_msgs::msg::PointField> pfs;
        pcl_conversions::fromPCL(m.fields, pfs);
        length += serializationLength(pfs); // fields
        length += 1; // is_bigendian
        length += 4; // point_step
        length += 4; // row_step
        length += 4; // data's size
        length += m.data.size() * sizeof(std::uint8_t);
        length += 1; // is_dense

        return length;
      }
    };
    **/
    /*
     * Provide a custom serialization for pcl::PCLPointField
     */
    /**
    template<>
    struct Serializer<pcl::PCLPointField>
    {
      template<typename Stream>
      inline static void write(Stream& stream, const pcl::PCLPointField& m)
      {
        stream.next(m.name);
        stream.next(m.offset);
        stream.next(m.datatype);
        stream.next(m.count);
      }

      template<typename Stream>
      inline static void read(Stream& stream, pcl::PCLPointField& m)
      {
        stream.next(m.name);
        stream.next(m.offset);
        stream.next(m.datatype);
        stream.next(m.count);
      }

      inline static uint32_t serializedLength(const pcl::PCLPointField& m)
      {
        uint32_t length = 0;

        length += serializationLength(m.name);
        length += serializationLength(m.offset);
        length += serializationLength(m.datatype);
        length += serializationLength(m.count);

        return length;
      }
    };
    **/
    /*
     * Provide a custom serialization for pcl::PCLHeader
     */
    /**
    template<>
    struct Serializer<pcl::PCLHeader>
    {
      template<typename Stream>
      inline static void write(Stream& stream, const pcl::PCLHeader& m)
      {
        std_msgs::msg::Header header;
        pcl_conversions::fromPCL(m, header);
        stream.next(header);
      }

      template<typename Stream>
      inline static void read(Stream& stream, pcl::PCLHeader& m)
      {
        std_msgs::msg::Header header;
        stream.next(header);
        pcl_conversions::toPCL(header, m);
      }

      inline static uint32_t serializedLength(const pcl::PCLHeader& m)
      {
        uint32_t length = 0;

        std_msgs::msg::Header header;
        pcl_conversions::fromPCL(m, header);
        length += serializationLength(header);

        return length;
      }
    };
  } // namespace ros::serialization
} // namespace ros
**/

#endif /* PCL_CONVERSIONS_H__ */
