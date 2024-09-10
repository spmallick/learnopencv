#include <string>

#include "gtest/gtest.h"

#include "pcl_conversions/pcl_conversions.h"

namespace {

class PCLConversionTests : public ::testing::Test {
protected:
  virtual void SetUp() {
    pcl_image.header.stamp = 3141592653;
    pcl_image.header.frame_id = "pcl";
    pcl_image.height = 1;
    pcl_image.width = 2;
    pcl_image.step = 1;
    pcl_image.is_bigendian = true;
    pcl_image.encoding = "bgr8";
    pcl_image.data.resize(2);
    pcl_image.data[0] = 0x42;
    pcl_image.data[1] = 0x43;

    pcl_pc2.header.stamp = 3141592653;
    pcl_pc2.header.frame_id = "pcl";
    pcl_pc2.height = 1;
    pcl_pc2.width = 2;
    pcl_pc2.point_step = 1;
    pcl_pc2.row_step = 1;
    pcl_pc2.is_bigendian = true;
    pcl_pc2.is_dense = true;
    pcl_pc2.fields.resize(2);
    pcl_pc2.fields[1].name = "XYZ";
    pcl_pc2.fields[1].datatype = pcl::PCLPointField::INT8;
    pcl_pc2.fields[1].count = 3;
    pcl_pc2.fields[1].offset = 0;
    pcl_pc2.fields[0].name = "RGB";
    pcl_pc2.fields[0].datatype = pcl::PCLPointField::INT8;
    pcl_pc2.fields[0].count = 3;
    pcl_pc2.fields[0].offset = 8 * 3;
    pcl_pc2.data.resize(2);
    pcl_pc2.data[0] = 0x42;
    pcl_pc2.data[1] = 0x43;
  }

  pcl::PCLImage pcl_image;
  sensor_msgs::msg::Image image;

  pcl::PCLPointCloud2 pcl_pc2;
  sensor_msgs::msg::PointCloud2 pc2;
};

template<class T>
void test_image(T &image) {
  EXPECT_EQ(std::string("pcl"), image.header.frame_id);
  EXPECT_EQ(1U, image.height);
  EXPECT_EQ(2U, image.width);
  EXPECT_EQ(1U, image.step);
  EXPECT_TRUE(image.is_bigendian);
  EXPECT_EQ(std::string("bgr8"), image.encoding);
  EXPECT_EQ(2U, image.data.size());
  EXPECT_EQ(0x42, image.data[0]);
  EXPECT_EQ(0x43, image.data[1]);
}

TEST_F(PCLConversionTests, imageConversion) {
  pcl_conversions::fromPCL(pcl_image, image);
  test_image(image);
  pcl::PCLImage pcl_image2;
  pcl_conversions::toPCL(image, pcl_image2);
  test_image(pcl_image2);
  EXPECT_EQ(pcl_image.header.stamp, pcl_image2.header.stamp);
}

template<class T>
void test_pc(T &pc) {
  EXPECT_EQ(std::string("pcl"), pc.header.frame_id);
  EXPECT_EQ(1U, pc.height);
  EXPECT_EQ(2U, pc.width);
  EXPECT_EQ(1U, pc.point_step);
  EXPECT_EQ(1U, pc.row_step);
  EXPECT_TRUE(pc.is_bigendian);
  EXPECT_TRUE(pc.is_dense);
  EXPECT_EQ("XYZ", pc.fields[0].name);
  EXPECT_EQ(pcl::PCLPointField::INT8, pc.fields[0].datatype);
  EXPECT_EQ(3U, pc.fields[0].count);
  EXPECT_EQ(0U, pc.fields[0].offset);
  EXPECT_EQ("RGB", pc.fields[1].name);
  EXPECT_EQ(pcl::PCLPointField::INT8, pc.fields[1].datatype);
  EXPECT_EQ(3U, pc.fields[1].count);
  EXPECT_EQ(8U * 3U, pc.fields[1].offset);
  EXPECT_EQ(2U, pc.data.size());
  EXPECT_EQ(0x42, pc.data[0]);
  EXPECT_EQ(0x43, pc.data[1]);
}

TEST_F(PCLConversionTests, pointcloud2Conversion) {
  pcl_conversions::fromPCL(pcl_pc2, pc2);
  test_pc(pc2);
  pcl::PCLPointCloud2 pcl_pc2_2;
  pcl_conversions::toPCL(pc2, pcl_pc2_2);
  test_pc(pcl_pc2_2);
  EXPECT_EQ(pcl_pc2.header.stamp, pcl_pc2_2.header.stamp);
}

} // namespace


struct StampTestData
{
  const rclcpp::Time stamp_;
  rclcpp::Time stamp2_;

  explicit StampTestData(const rclcpp::Time &stamp)
    : stamp_(stamp)
  {
    std::uint64_t pcl_stamp;
    pcl_conversions::toPCL(stamp_, pcl_stamp);
    pcl_conversions::fromPCL(pcl_stamp, stamp2_);
  }
};

TEST(PCLConversionStamp, Stamps)
{
  {
    const StampTestData d(rclcpp::Time(1, 1000));
    EXPECT_TRUE(d.stamp_==d.stamp2_);
  }

  {
    const StampTestData d(rclcpp::Time(1, 999999000));
    EXPECT_TRUE(d.stamp_==d.stamp2_);
  }

  {
    const StampTestData d(rclcpp::Time(1, 999000000));
    EXPECT_TRUE(d.stamp_==d.stamp2_);
  }

  {
    const StampTestData d(rclcpp::Time(1423680574, 746000000));
    EXPECT_TRUE(d.stamp_==d.stamp2_);
  }

  {
    const StampTestData d(rclcpp::Time(1423680629, 901000000));
    EXPECT_TRUE(d.stamp_==d.stamp2_);
  }
}

int main(int argc, char **argv) {
  try {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  } catch (std::exception &e) {
    std::cerr << "Unhandled Exception: " << e.what() << std::endl;
  }
  return 1;
}
