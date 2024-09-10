^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package pcl_ros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.6.2 (2018-05-20)
------------------
* Fix exported includes in Ubuntu Artful
* Increase limits on CropBox filter parameters
* Contributors: James Ward, Jiri Horner

1.6.1 (2018-05-08)
------------------
* Add 1.6.0 section to CHANGELOG.rst
* Fix the use of Eigen3 in cmake
* Contributors: Kentaro Wada

1.6.0 (2018-04-30)
------------------

* Fix build and update maintainers
* Add message_filters to find_package
* Remove unnecessary dependency on genmsg
* Contributors: Paul Bovbel, Kentaro Wada

1.5.4 (2018-03-31)
------------------
* update to use non deprecated pluginlib macro
* Fix config path of sample_voxel_grid.launch
* remove hack now that upstream pcl has been rebuilt
* Looser hzerror in test for extract_clusters to make it pass on Travis
* Add sample & test for surface/convex_hull
* Add sample & test for segmentation/extract_clusters.cpp
* Add sample & test for io/concatenate_data.cpp
* Add sample & test for features/normal_3d.cpp
* Organize samples of pcl_ros/features
* Add test arg to avoid duplicated testing
* LazyNodelet for features/*
* LazyNodelet for filters/ProjectInliers
* Refactor io/PCDReader and io/PCDWriter as child of PCLNodelet
* LazyNodelet for io/PointCloudConcatenateFieldsSynchronizer
* LazyNodelet for io/PointCloudConcatenateDataSynchronizer
* LazyNodelet for segmentation/SegmentDifferences
* LazyNodelet for segmentation/SACSegmentationFromNormals
* LazyNodelet for segmentation/SACSegmentation
* LazyNodelet for segmentation/ExtractPolygonalPrismData
* LazyNodelet for segmentation/EuclideanClusterExtraction
* LazyNodelet for surface/MovingLeastSquares
* LazyNodelet for surface/ConvexHull2D
* Add missing COMPONENTS of PCL
* Inherit NodeletLazy for pipeline with less cpu load
* Set leaf_size 0.02
* Install samples
* Add sample and test for pcl/StatisticalOutlierRemoval
  Conflicts:
  pcl_ros/CMakeLists.txt
* Add sample and test for pcl/VoxelGrid
  Conflicts:
  pcl_ros/CMakeLists.txt
* no need to remove duplicates
* spourious line change
* remove now unnecessary build_depend on qtbase5
* exclude PCL IO libraries exporting Qt flag
* find only PCL components used instead of all PCL
* Remove dependency on vtk/libproj-dev (`#145 <https://github.com/ros-perception/perception_pcl/issues/145>`_)
  * Remove dependency on vtk/libproj-dev
  These dependencies were introduced in `#124 <https://github.com/ros-perception/perception_pcl/issues/124>`_ to temporarily fix
  missing / wrong dependencies in upstream vtk. This hack is no longer
  necessary, since fixed vtk packages have been uploaded to
  packages.ros.org (see `#124 <https://github.com/ros-perception/perception_pcl/issues/124>`_ and `ros-infrastructure/reprepro-updater#32 <https://github.com/ros-infrastructure/reprepro-updater/issues/32>`_).
  * Remove vtk hack from CMakeLists.txt
* Contributors: Kentaro Wada, Mikael Arguedas

1.5.3 (2017-05-03)
------------------
* Add dependency on qtbase5-dev for find_package(Qt5Widgets)
  See https://github.com/ros-perception/perception_pcl/pull/117#issuecomment-298158272 for detail.
* Contributors: Kentaro Wada

1.5.2 (2017-04-29)
------------------
* Find Qt5Widgets to fix -lQt5::Widgets error
* Contributors: Kentaro Wada

1.5.1 (2017-04-26)
------------------
* Add my name as a maintainer
* Contributors: Kentaro Wada

1.5.0 (2017-04-25)
------------------
* Fix lib name duplication error in ubunt:zesty
* Detect automatically the version of PCL in cmake
* Install xml files declaring nodelets
* Fix syntax of nodelet manifest file by splitting files for each library.
* Contributors: Kentaro Wada

1.4.0 (2016-04-22)
------------------
* Fixup libproj-dev rosdep
* Add build depend on libproj, since it's not provided by vtk right now
* manually remove dependency on vtkproj from PCL_LIBRARIES
* Remove python-vtk for kinetic-devel, see issue `#44 <https://github.com/ros-perception/perception_pcl/issues/44>`_
* Contributors: Jackie Kay, Paul Bovbel

1.3.0 (2015-06-22)
------------------
* cleanup broken library links
  All removed library names are included in ${PCL_LIBRARIES}.
  However, the plain library names broke catkin's overlay mechanism:
  Where ${PCL_LIBRARIES} could point to a local installation of the PCL,
  e.g. pcd_ros_segmentation might still link to the system-wide installed version
  of pcl_segmentation.
* Fixed test for jade-devel. Progress on `#92 <https://github.com/ros-perception/perception_pcl/issues/92>`_
* commented out test_tf_message_filter_pcl
  Until `ros/geometry#80 <https://github.com/ros/geometry/issues/80>`_ has been merged the test will fail.
* fixed indentation and author
* Adds a test for tf message filters with pcl pointclouds
* specialized HasHeader, TimeStamp, FrameId
  - HasHeader now returns false
  - TimeStamp and FrameId specialed for pcl::PointCloud<T> for any point type T
  These changes allow to use pcl::PointCloud with tf::MessageFilter
* Sync pcl_nodelets.xml from hydro to indigo
  Fixes to pass catkin lint -W1
* Fixes `#87 <https://github.com/ros-perception/perception_pcl/issues/87>`_ for Indigo
* Fixes `#85 <https://github.com/ros-perception/perception_pcl/issues/85>`_ for Indigo
* Fixes `#77 <https://github.com/ros-perception/perception_pcl/issues/77>`_ and `#80 <https://github.com/ros-perception/perception_pcl/issues/80>`_ for indigo
* Added option to save pointclouds in binary and binary compressed format
* Contributors: Brice Rebsamen, Lucid One, Mitchell Wills, v4hn

1.2.6 (2015-02-04)
------------------

1.2.5 (2015-01-20)
------------------

1.2.4 (2015-01-15)
------------------

1.2.3 (2015-01-10)
------------------
* Update common.py
  Extended filter limits up to ±100000.0 in order to support intensity channel filtering.
* Contributors: Dani Carbonell

1.2.2 (2014-10-25)
------------------
* Adding target_frame
  [Ability to specify frame in bag_to_pcd ](https://github.com/ros-perception/perception_pcl/issues/55)
* Update pcl_nodelets.xml
  Included missing closing library tag.  This was causing the pcl/Filter nodelets below the missing nodelet tag to not be exported correctly.
* Contributors: Matt Derry, Paul Bovbel, Ruffin

1.2.1 (2014-09-13)
------------------
* clean up merge
* merge pull request `#60 <https://github.com/ros-perception/perception_pcl/issues/60>`_
* Contributors: Paul Bovbel

1.2.0 (2014-04-09)
------------------
* Updated maintainership
* Fix TF2 support for bag_to_pcd `#46 <https://github.com/ros-perception/perception_pcl/issues/46>`_
* Use cmake_modules to find eigen on indigo `#45 <https://github.com/ros-perception/perception_pcl/issues/45>`_

1.1.7 (2013-09-20)
------------------
* adding more uncaught config dependencies
* adding FeatureConfig dependency too

1.1.6 (2013-09-20)
------------------
* add excplicit dependency on gencfg target

1.1.5 (2013-08-27)
------------------
* Updated package.xml's to use new libpcl-all rosdep rules
* package.xml: tuned whitespaces
  This commit removes trailing whitespaces and makes the line with the license information in the package.xml bitwise match exactly the common license information line in most ROS packages.
  The trailing whitespaces were detected when providing a bitbake recipe in the meta-ros project (github.com/bmwcarit/meta-ros). In the recipe, the hash of the license line is declared and is used to check for changes in the license. For this recipe, it was not matching the common one.
  A related already merged commit is https://github.com/ros/std_msgs/pull/3 and a related pending commit is https://github.com/ros-perception/pcl_msgs/pull/1.

1.1.4 (2013-07-23)
------------------
* Fix a serialization error with point_cloud headers
* Initialize shared pointers before use in part of the pcl_conversions
  Should address runtime errors reported in `#29 <https://github.com/ros-perception/perception_pcl/issues/29>`_
* Changed the default bounds on filters to -1000, 1000 from -5, 5 in common.py

1.1.2 (2013-07-19)
------------------
* Fixed missing package exports on pcl_conversions and others
* Make find_package on Eigen and PCL REQUIRED

1.1.1 (2013-07-10)
------------------
* Add missing EIGEN define which caused failures on the farm

1.1.0 (2013-07-09)
------------------
* Add missing include in one of the installed headers
* Refactors to use pcl-1.7
* Use the PointIndices from pcl_msgs
* Experimental changes to point_cloud.h
* Fixes from converting from pcl-1.7, incomplete
* Depend on pcl_conversions and pcl_msgs
* bag_to_pcd: check return code of transformPointCloud()
  This fixes a bug where bag_to_pcd segfaults because of an ignored
  tf::ExtrapolationException.
* Changed #include type to lib
* Changed some #include types to lib
* removed a whitespace

1.0.34 (2013-05-21)
-------------------
* fixing catkin python imports

1.0.33 (2013-05-20)
-------------------
* Fixing catkin python imports

1.0.32 (2013-05-17)
-------------------
* Merge pull request `#11 <https://github.com/ros-perception/perception_pcl/issues/11>`_ from k-okada/groovy-devel
  revert removed directories
* fix to compileable
* copy features/segmentation/surface from fuerte-devel

1.0.31 (2013-04-22 11:58)
-------------------------
* No changes

1.0.30 (2013-04-22 11:47)
-------------------------
* deprecating bin install targets

1.0.29 (2013-03-04)
-------------------
* Fixes `#7 <https://github.com/ros-perception/perception_pcl/issues/7>`_
* now also works without specifying publishing interval like described in the wiki.

1.0.28 (2013-02-05 12:29)
-------------------------
* reenabling deprecated install targets - comment added

1.0.27 (2013-02-05 12:10)
-------------------------
* Update pcl_ros/package.xml
* Fixing target install directory for pcl tools
* update pluginlib macro

1.0.26 (2013-01-17)
-------------------
* fixing catkin export

1.0.25 (2013-01-01)
-------------------
* fixes `#1 <https://github.com/ros-perception/perception_pcl/issues/1>`_

1.0.24 (2012-12-21)
-------------------
* remove obsolete roslib import

1.0.23 (2012-12-19 16:52)
-------------------------
* clean up shared parameters

1.0.22 (2012-12-19 15:22)
-------------------------
* fix dyn reconf files

1.0.21 (2012-12-18 17:42)
-------------------------
* fixing catkin_package debs

1.0.20 (2012-12-18 14:21)
-------------------------
* adding catkin_project dependencies

1.0.19 (2012-12-17 21:47)
-------------------------
* adding nodelet_topic_tools dependency

1.0.18 (2012-12-17 21:17)
-------------------------
* adding pluginlib dependency
* adding nodelet dependencies
* CMake install fixes
* migrating nodelets and tools from fuerte release to pcl_ros
* Updated for new <buildtool_depend>catkin<...> catkin rule

1.0.17 (2012-10-26 09:28)
-------------------------
* remove useless tags

1.0.16 (2012-10-26 08:53)
-------------------------
* no need to depend on a meta-package

1.0.15 (2012-10-24)
-------------------
* do not generrate messages automatically

1.0.14 (2012-10-23)
-------------------
* bring back the PCL msgs

1.0.13 (2012-10-11 17:46)
-------------------------
* install library to the right place

1.0.12 (2012-10-11 17:25)
-------------------------

1.0.11 (2012-10-10)
-------------------
* fix a few dependencies

1.0.10 (2012-10-04)
-------------------
* comply to the new catkin API
* fixed pcl_ros manifest
* added pcl exports in manifest.xml
* fixed rosdeb pcl in pcl_ros/manifest.xml
* removing common_rosdeps from manifest.xml
* perception_pcl restructuring in groovy branch
* restructuring perception_pcl in groovy branch
* catkinized version of perception_pcl for groovy
* added PCL 1.6 stack for groovy
