/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PCL_ROS__POINT_CLOUD_HPP__
#define PCL_ROS__POINT_CLOUD_HPP__

// test if testing machinery can be implemented
#if defined(__cpp_rvalue_references) && defined(__cpp_constexpr)
#define ROS_POINTER_COMPATIBILITY_IMPLEMENTED 1
#else
#define ROS_POINTER_COMPATIBILITY_IMPLEMENTED 0
#endif

#include <pcl/point_cloud.h>
#include <pcl/pcl_config.h>  // for PCL_VERSION_COMPARE
#if PCL_VERSION_COMPARE(>=, 1, 11, 0)
#include <pcl/type_traits.h>
#else
#include <pcl/point_traits.h>
#endif  // PCL_VERSION_COMPARE(>=, 1, 11, 0)
#include <pcl/for_each_type.h>
#include <pcl/conversions.h>
#if ROS_POINTER_COMPATIBILITY_IMPLEMENTED
#if PCL_VERSION_COMPARE(>=, 1, 11, 0)
#include <pcl/memory.h>
#elif PCL_VERSION_COMPARE(>=, 1, 10, 0)
#include <pcl/make_shared.h>
#endif
#endif
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <utility>
#include <vector>
#if ROS_POINTER_COMPATIBILITY_IMPLEMENTED
#include <type_traits>
#include <memory>
#endif
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <boost/foreach.hpp>  // for BOOST_FOREACH
#include <boost/mpl/size.hpp>
#include <boost/ref.hpp>

namespace pcl
{
namespace detail
{
template<typename Stream, typename PointT>
struct FieldStreamer
{
  explicit FieldStreamer(Stream & stream)
  : stream_(stream) {}

  template<typename U>
  void operator()()
  {
    const char * name = pcl::traits::name<PointT, U>::value;
    std::uint32_t name_length = strlen(name);
    stream_.next(name_length);
    if (name_length > 0) {
      memcpy(stream_.advance(name_length), name, name_length);
    }

    std::uint32_t offset = pcl::traits::offset<PointT, U>::value;
    stream_.next(offset);

    std::uint8_t datatype = pcl::traits::datatype<PointT, U>::value;
    stream_.next(datatype);

    std::uint32_t count = pcl::traits::datatype<PointT, U>::size;
    stream_.next(count);
  }

  Stream & stream_;
};

template<typename PointT>
struct FieldsLength
{
  FieldsLength()
  : length(0) {}

  template<typename U>
  void operator()()
  {
    std::uint32_t name_length = strlen(pcl::traits::name<PointT, U>::value);
    length += name_length + 13;
  }

  std::uint32_t length;
};
}  // namespace detail
}  // namespace pcl

namespace ros
{
// In ROS 1.3.1+, we can specialize the functor used to create PointCloud<T> objects
// on the subscriber side. This allows us to generate the mapping between message
// data and object fields only once and reuse it.
#if 0  // ROS_VERSION_MINIMUM(1, 3, 1)
template<typename T>
struct DefaultMessageCreator<pcl::PointCloud<T>>
{
  boost::shared_ptr<pcl::MsgFieldMap> mapping_;

  DefaultMessageCreator()
  : mapping_(boost::make_shared<pcl::MsgFieldMap>() )
  {
  }

  boost::shared_ptr<pcl::PointCloud<T>> operator()()
  {
    boost::shared_ptr<pcl::PointCloud<T>> msg(new pcl::PointCloud<T>());
    pcl::detail::getMapping(*msg) = mapping_;
    return msg;
  }
};
#endif

namespace message_traits
{
template<typename T>
struct MD5Sum<pcl::PointCloud<T>>
{
  static const char * value() {return MD5Sum<sensor_msgs::PointCloud2>::value();}
  static const char * value(const pcl::PointCloud<T> &) {return value();}

  static const uint64_t static_value1 = MD5Sum<sensor_msgs::PointCloud2>::static_value1;
  static const uint64_t static_value2 = MD5Sum<sensor_msgs::PointCloud2>::static_value2;

  // If the definition of sensor_msgs/PointCloud2 changes, we'll get a compile error here.
  ROS_STATIC_ASSERT(static_value1 == 0x1158d486dd51d683ULL);
  ROS_STATIC_ASSERT(static_value2 == 0xce2f1be655c3c181ULL);
};

template<typename T>
struct DataType<pcl::PointCloud<T>>
{
  static const char * value() {return DataType<sensor_msgs::PointCloud2>::value();}
  static const char * value(const pcl::PointCloud<T> &) {return value();}
};

template<typename T>
struct Definition<pcl::PointCloud<T>>
{
  static const char * value() {return Definition<sensor_msgs::PointCloud2>::value();}
  static const char * value(const pcl::PointCloud<T> &) {return value();}
};

// pcl point clouds message don't have a ROS compatible header
// the specialized meta functions below (TimeStamp and FrameId)
// can be used to get the header data.
template<typename T>
struct HasHeader<pcl::PointCloud<T>>: FalseType {};

template<typename T>
struct TimeStamp<pcl::PointCloud<T>>
{
  // This specialization could be dangerous, but it's the best I can do.
  // If this TimeStamp struct is destroyed before they are done with the
  // pointer returned by the first functions may go out of scope, but there
  // isn't a lot I can do about that. This is a good reason to refuse to
  // returning pointers like this...
  static ros::Time * pointer(typename pcl::PointCloud<T> & m)
  {
    header_.reset(new std_msgs::Header());
    pcl_conversions::fromPCL(m.header, *(header_));
    return &(header_->stamp);
  }
  static ros::Time const * pointer(const typename pcl::PointCloud<T> & m)
  {
    header_const_.reset(new std_msgs::Header());
    pcl_conversions::fromPCL(m.header, *(header_const_));
    return &(header_const_->stamp);
  }
  static ros::Time value(const typename pcl::PointCloud<T> & m)
  {
    return pcl_conversions::fromPCL(m.header).stamp;
  }

private:
  static boost::shared_ptr<std_msgs::Header> header_;
  static boost::shared_ptr<std_msgs::Header> header_const_;
};

template<typename T>
struct FrameId<pcl::PointCloud<T>>
{
  static std::string * pointer(pcl::PointCloud<T> & m) {return &m.header.frame_id;}
  static std::string const * pointer(const pcl::PointCloud<T> & m) {return &m.header.frame_id;}
  static std::string value(const pcl::PointCloud<T> & m) {return m.header.frame_id;}
};

}  // namespace message_traits

namespace serialization
{
template<typename T>
struct Serializer<pcl::PointCloud<T>>
{
  template<typename Stream>
  inline static void write(Stream & stream, const pcl::PointCloud<T> & m)
  {
    stream.next(m.header);

    // Ease the user's burden on specifying width/height for unorganized datasets
    uint32_t height = m.height, width = m.width;
    if (height == 0 && width == 0) {
      width = m.points.size();
      height = 1;
    }
    stream.next(height);
    stream.next(width);

    // Stream out point field metadata
    typedef typename pcl::traits::fieldList<T>::type FieldList;
    uint32_t fields_size = boost::mpl::size<FieldList>::value;
    stream.next(fields_size);
    pcl::for_each_type<FieldList>(pcl::detail::FieldStreamer<Stream, T>(stream));

    // Assume little-endian...
    uint8_t is_bigendian = false;
    stream.next(is_bigendian);

    // Write out point data as binary blob
    uint32_t point_step = sizeof(T);
    stream.next(point_step);
    uint32_t row_step = point_step * width;
    stream.next(row_step);
    uint32_t data_size = row_step * height;
    stream.next(data_size);
    memcpy(stream.advance(data_size), m.points.data(), data_size);

    uint8_t is_dense = m.is_dense;
    stream.next(is_dense);
  }

  template<typename Stream>
  inline static void read(Stream & stream, pcl::PointCloud<T> & m)
  {
    std_msgs::Header header;
    stream.next(header);
    pcl_conversions::toPCL(header, m.header);
    stream.next(m.height);
    stream.next(m.width);

    /// @todo Check that fields haven't changed!
    std::vector<sensor_msgs::PointField> fields;
    stream.next(fields);

    // Construct field mapping if deserializing for the first time
    boost::shared_ptr<pcl::MsgFieldMap> & mapping_ptr = pcl::detail::getMapping(m);
    if (!mapping_ptr) {
      // This normally should get allocated by DefaultMessageCreator, but just in case
      mapping_ptr = boost::make_shared<pcl::MsgFieldMap>();
    }
    pcl::MsgFieldMap & mapping = *mapping_ptr;
    if (mapping.empty()) {
      pcl::createMapping<T>(fields, mapping);
    }

    uint8_t is_bigendian;
    stream.next(is_bigendian);     // ignoring...
    uint32_t point_step, row_step;
    stream.next(point_step);
    stream.next(row_step);

    // Copy point data
    uint32_t data_size;
    stream.next(data_size);
    assert(data_size == m.height * m.width * point_step);
    m.points.resize(m.height * m.width);
    uint8_t * m_data = reinterpret_cast<uint8_t *>(m.points.data());
    // If the data layouts match, can copy a whole row in one memcpy
    if (mapping.size() == 1 &&
      mapping[0].serialized_offset == 0 &&
      mapping[0].struct_offset == 0 &&
      point_step == sizeof(T))
    {
      uint32_t m_row_step = sizeof(T) * m.width;
      // And if the row steps match, can copy whole point cloud in one memcpy
      if (m_row_step == row_step) {
        memcpy(m_data, stream.advance(data_size), data_size);
      } else {
        for (uint32_t i = 0; i < m.height; ++i, m_data += m_row_step) {
          memcpy(m_data, stream.advance(row_step), m_row_step);
        }
      }
    } else {
      // If not, do a lot of memcpys to copy over the fields
      for (uint32_t row = 0; row < m.height; ++row) {
        const uint8_t * stream_data = stream.advance(row_step);
        for (uint32_t col = 0; col < m.width; ++col, stream_data += point_step) {
          BOOST_FOREACH(const pcl::detail::FieldMapping & fm, mapping) {
            memcpy(m_data + fm.struct_offset, stream_data + fm.serialized_offset, fm.size);
          }
          m_data += sizeof(T);
        }
      }
    }

    uint8_t is_dense;
    stream.next(is_dense);
    m.is_dense = is_dense;
  }

  inline static uint32_t serializedLength(const pcl::PointCloud<T> & m)
  {
    uint32_t length = 0;

    length += serializationLength(m.header);
    length += 8;     // height/width

    pcl::detail::FieldsLength<T> fl;
    typedef typename pcl::traits::fieldList<T>::type FieldList;
    pcl::for_each_type<FieldList>(boost::ref(fl));
    length += 4;     // size of 'fields'
    length += fl.length;

    length += 1;     // is_bigendian
    length += 4;     // point_step
    length += 4;     // row_step
    length += 4;     // size of 'data'
    length += m.points.size() * sizeof(T);     // data
    length += 1;     // is_dense

    return length;
  }
};
}  // namespace serialization

/// @todo Printer specialization in message_operations

}  // namespace ros

namespace pcl
{
namespace detail
{
#if ROS_POINTER_COMPATIBILITY_IMPLEMENTED
#if PCL_VERSION_COMPARE(>=, 1, 10, 0)
template<class T>
constexpr static bool pcl_uses_boost = std::is_same<boost::shared_ptr<T>,
    pcl::shared_ptr<T>>::value;
#else
template<class T>
constexpr static bool pcl_uses_boost = true;
#endif

template<class SharedPointer>
struct Holder
{
  SharedPointer p;

  explicit Holder(const SharedPointer & p)
  : p(p) {}
  Holder(const Holder & other)
  : p(other.p) {}
  Holder(Holder && other)
  : p(std::move(other.p)) {}

  void operator()(...) {p.reset();}
};

template<class T>
inline std::shared_ptr<T> to_std_ptr(const boost::shared_ptr<T> & p)
{
  typedef Holder<std::shared_ptr<T>> H;
  if (H * h = boost::get_deleter<H>(p)) {
    return h->p;
  } else {
    return std::shared_ptr<T>(p.get(), Holder<boost::shared_ptr<T>>(p));
  }
}

template<class T>
inline boost::shared_ptr<T> to_boost_ptr(const std::shared_ptr<T> & p)
{
  typedef Holder<boost::shared_ptr<T>> H;
  if (H * h = std::get_deleter<H>(p)) {
    return h->p;
  } else {
    return boost::shared_ptr<T>(p.get(), Holder<std::shared_ptr<T>>(p));
  }
}
#endif
}  // namespace detail

// add functions to convert to smart pointer used by ROS
template<class T>
inline boost::shared_ptr<T> ros_ptr(const boost::shared_ptr<T> & p)
{
  return p;
}

#if ROS_POINTER_COMPATIBILITY_IMPLEMENTED
template<class T>
inline boost::shared_ptr<T> ros_ptr(const std::shared_ptr<T> & p)
{
  return detail::to_boost_ptr(p);
}

// add functions to convert to smart pointer used by PCL, based on PCL's own pointer
template<class T, class = typename std::enable_if<!detail::pcl_uses_boost<T>>::type>
inline std::shared_ptr<T> pcl_ptr(const std::shared_ptr<T> & p)
{
  return p;
}

template<class T, class = typename std::enable_if<!detail::pcl_uses_boost<T>>::type>
inline std::shared_ptr<T> pcl_ptr(const boost::shared_ptr<T> & p)
{
  return detail::to_std_ptr(p);
}

template<class T, class = typename std::enable_if<detail::pcl_uses_boost<T>>::type>
inline boost::shared_ptr<T> pcl_ptr(const std::shared_ptr<T> & p)
{
  return detail::to_boost_ptr(p);
}

template<class T, class = typename std::enable_if<detail::pcl_uses_boost<T>>::type>
inline boost::shared_ptr<T> pcl_ptr(const boost::shared_ptr<T> & p)
{
  return p;
}
#else
template<class T>
inline boost::shared_ptr<T> pcl_ptr(const boost::shared_ptr<T> & p)
{
  return p;
}
#endif
}  // namespace pcl

#endif  // PCL_ROS__POINT_CLOUD_HPP__
