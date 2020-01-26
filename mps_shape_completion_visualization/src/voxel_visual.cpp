/*
 * Copyright (c) 2012, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <OGRE/OgreVector3.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreSceneManager.h>

#include <rviz/ogre_helpers/point_cloud.h>
#include <std_msgs/Float32MultiArray.h>

#include "voxel_visual.h"

namespace mps_shape_completion_visualization
{

// BEGIN_TUTORIAL
    VoxelGridVisual::VoxelGridVisual( Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node )
    {
        scene_manager_ = scene_manager;

        // Ogre::SceneNode s form a tree, with each node storing the
        // transform (position and orientation) of itself relative to its
        // parent.  Ogre does the math of combining those transforms when it
        // is time to render.
        //
        // Here we create a node to store the pose of the Imu's header frame
        // relative to the RViz fixed frame.
        frame_node_ = parent_node->createChildSceneNode();

        // We create the arrow object within the frame node so that we can
        // set its position and direction relative to its header frame.
        // voxel_grid_.reset(new rviz::PointCloud( scene_manager_, frame_node_ ));
        voxel_grid_.reset(new rviz::PointCloud());
        voxel_grid_->setRenderMode(rviz::PointCloud::RM_BOXES);
        frame_node_->attachObject(voxel_grid_.get());
    }

    VoxelGridVisual::~VoxelGridVisual()
    {
        // Destroy the frame node since we don't need it anymore.
        scene_manager_->destroySceneNode( frame_node_ );
    }

    void VoxelGridVisual::setMessage( const mps_shape_completion_msgs::OccupancyStamped::ConstPtr& msg)
    {
        latest_msg = *msg;
        updatePointCloud();
    }

    void VoxelGridVisual::updatePointCloud()
    {
        if(latest_msg.occupancy.layout.dim.size() == 0)
        {
            return;
        }
        
        double scale = latest_msg.scale;
        voxel_grid_->setDimensions(scale, scale, scale);


        const std::vector<float> data = latest_msg.occupancy.data;
        const std::vector<std_msgs::MultiArrayDimension> dims = latest_msg.occupancy.layout.dim;
        int data_offset = latest_msg.occupancy.layout.data_offset;

        std::vector< rviz::PointCloud::Point> points;
        for(int i=0; i<dims[0].size; i++)
        {
            for(int j=0; j<dims[1].size; j++)
            {
                for(int k=0; k<dims[2].size; k++)
                {
                    float val = data[data_offset + dims[1].stride * i + dims[2].stride * j + k];
                    if(val < threshold_)
                    {
                        continue;
                    }
                    
                    rviz::PointCloud::Point p;
                    p.position.x = scale/2 + i*scale;
                    p.position.y = scale/2 + j*scale;
                    p.position.z = scale/2 + k*scale;

                    if(p.position.x > 0.2)
                    {
                        val = 0.1;
                    }


                    if(binary_display_)
                    {
                        val = 1.0;
                    }

                    p.setColor(r_, g_, b_, val*a_);
                    
                    points.push_back(p);
                }
            }
        }
        
        voxel_grid_->clear();

        //this has a per-point setting, but setting it
        bool use_per_point = !(a_ >= 1.0 && binary_display_);
        voxel_grid_->setAlpha(a_, use_per_point); 
        
        voxel_grid_->addPoints(&points.front(), points.size());

        std::cout << "Added " << points.size() << " points.\n";
        std::cout << "Use global alpha is " << binary_display_ << "\n";
        std::cout << "global alpha is " << a_ << "\n";
        std::cout << "threshold is " << threshold_ << "\n";
    }



// Position and orientation are passed through to the SceneNode.
    void VoxelGridVisual::setFramePosition( const Ogre::Vector3& position )
    {
        frame_node_->setPosition( position );
    }

    void VoxelGridVisual::setFrameOrientation( const Ogre::Quaternion& orientation )
    {
        frame_node_->setOrientation( orientation );
    }

// Color is passed through to the Arrow object.
    void VoxelGridVisual::setColor( float r, float g, float b, float a )
    {
        r_ = r;
        g_ = g;
        b_ = b;
        a_ = a;
        // voxel_grid_->setColor( r, g, b, a );
    }

    void VoxelGridVisual::setBinaryDisplay(bool binary_display)
    {
        binary_display_ = binary_display;
    }

    void VoxelGridVisual::setThreshold(float threshold)
    {
        threshold_ = threshold;
    }
// END_TUTORIAL

} // end namespace mps_shape_completion_visualization
