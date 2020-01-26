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

#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreSceneManager.h>

#include <tf/transform_listener.h>

#include <rviz/visualization_manager.h>
#include <rviz/properties/color_property.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/int_property.h>
#include <rviz/frame_manager.h>

#include "voxel_visual.h"

#include "voxel_display.h"

namespace mps_shape_completion_visualization
{

// BEGIN_TUTORIAL
// The constructor must have no arguments, so we can't give the
// constructor the parameters it needs to fully initialize.
    VoxelGridDisplay::VoxelGridDisplay()
    {
        color_property_ = new rviz::ColorProperty( "Color", QColor( 204, 51, 204 ),
                                                   "Color to draw the acceleration arrows.",
                                                   this, SLOT( updateColorAndAlpha() ));

        alpha_property_ = new rviz::FloatProperty( "Alpha Multiple", 1.0,
                                                   "0 is fully transparent, 1.0 is fully opaque.",
                                                   this, SLOT( updateColorAndAlpha() ));


        binary_display_property_ = new rviz::BoolProperty("Binary Display", true,
                                                            "If checked, all voxels will have the same alpha", this, SLOT(updateColorAndAlpha() ));

        cutoff_property_ = new rviz::FloatProperty("Threshold", 0.5,
                                                   "Voxels with values less than this will not be displayed",
                                                   this, SLOT(updateColorAndAlpha() ));


    }

// After the top-level rviz::Display::initialize() does its own setup,
// it calls the subclass's onInitialize() function.  This is where we
// instantiate all the workings of the class.  We make sure to also
// call our immediate super-class's onInitialize() function, since it
// does important stuff setting up the message filter.
//
//  Note that "MFDClass" is a typedef of
// ``MessageFilterDisplay<message type>``, to save typing that long
// templated class name every time you need to refer to the
// superclass.
    void VoxelGridDisplay::onInitialize()
    {
        MFDClass::onInitialize();
        visual_.reset(new VoxelGridVisual( context_->getSceneManager(), scene_node_ ));
        updateColorAndAlpha();
    }

    VoxelGridDisplay::~VoxelGridDisplay()
    {
    }

    void VoxelGridDisplay::reset()
    {
        MFDClass::reset();
    }

// Set the current color and alpha values for each visual.
    void VoxelGridDisplay::updateColorAndAlpha()
    {
        float alpha = alpha_property_->getFloat();
        Ogre::ColourValue color = color_property_->getOgreColor();
        visual_->setBinaryDisplay(binary_display_property_->getBool());
        visual_->setColor( color.r, color.g, color.b, alpha );
        std::cout << "Getting cutoff property: " << cutoff_property_->getFloat() << "\n";
        visual_->setThreshold(cutoff_property_->getFloat());
        visual_->updatePointCloud();
    }


// This is our callback to handle an incoming message.
    void VoxelGridDisplay::processMessage( const mps_shape_completion_msgs::OccupancyStamped::ConstPtr& msg)
    {
        // Here we call the rviz::FrameManager to get the transform from the
        // fixed frame to the frame in the header of this Imu message.  If
        // it fails, we can't do anything else so we return.
        Ogre::Quaternion orientation;
        Ogre::Vector3 position;
        if( !context_->getFrameManager()->getTransform( msg->header.frame_id,
                                                        msg->header.stamp,
                                                        position, orientation ))
        {
            ROS_DEBUG( "Error transforming from frame '%s' to frame '%s'",
                       msg->header.frame_id.c_str(), qPrintable( fixed_frame_ ));
            return;
        }


        // Now set or update the contents of the chosen visual.
        visual_->setMessage( msg );
        visual_->setFramePosition( position );
        visual_->setFrameOrientation( orientation );

        // float alpha = alpha_property_->getFloat();
        // Ogre::ColourValue color = color_property_->getOgreColor();
        // visual_->setColor( color.r, color.g, color.b, alpha );

    }


    
} // end namespace mps_shape_completion_visualization



// Tell pluginlib about this class.  It is important to do this in
// global scope, outside our package's namespace.
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mps_shape_completion_visualization::VoxelGridDisplay, rviz::Display )

