/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * Blind Spot Dataset Module
 * 
 * Provides dataset generation functions that apply blind spot masking
 * to demonstrate how neural networks learn to predict missing visual information.
 */

import {Example2D} from "./dataset";
import {VisionBlindSpot} from "./vision-blindspot";

/**
 * Configuration for blind spot dataset generation
 */
export interface BlindSpotDatasetConfig {
  blindSpot: VisionBlindSpot;
  baseDatasetGenerator: (numSamples: number, noiseLevel: number) => Example2D[];
}

/**
 * Apply blind spot masking to a dataset
 * Returns a dataset where points in the blind spot region have their labels masked
 */
export function applyBlindSpotToDataset(
  config: BlindSpotDatasetConfig,
  numSamples: number,
  noiseLevel: number
): Example2D[] {
  // Generate base dataset
  const baseData = config.baseDatasetGenerator(numSamples, noiseLevel);
  
  // Apply blind spot masking
  return baseData.map(point => ({
    x: point.x,
    y: point.y,
    label: config.blindSpot.isInBlindSpot(point.x, point.y) 
      ? 0  // Masked: neutral value
      : point.label
  }));
}

/**
 * Create a dataset with blind spot where network must learn to predict missing regions
 * The blind spot is visible but data is masked - network learns context from surroundings
 */
export function createContextLearningDataset(
  config: BlindSpotDatasetConfig,
  numSamples: number,
  noiseLevel: number
): Example2D[] {
  const baseData = config.baseDatasetGenerator(numSamples, noiseLevel);
  
  return baseData.map(point => {
    const mask = config.blindSpot.getBlindSpotMask(point.x, point.y);
    
    // Smoothly mask the label based on distance from blind spot center
    return {
      x: point.x,
      y: point.y,
      label: point.label * (1 - mask)
    };
  });
}

/**
 * Create a predictive dataset where the network learns to infer the blind spot content
 * Points are included in training with masked labels; network learns patterns
 */
export function createPredictiveDataset(
  config: BlindSpotDatasetConfig,
  numSamples: number,
  noiseLevel: number
): Example2D[] {
  const baseData = config.baseDatasetGenerator(numSamples, noiseLevel);
  
  return baseData.map(point => {
    if (config.blindSpot.isInBlindSpot(point.x, point.y)) {
      // For blind spot region, predict based on surrounding context
      const predictedLabel = config.blindSpot.predictBlindSpotValue(
        point.x,
        point.y,
        (x, y) => {
          // Find closest point in base data and return its label
          let closest = baseData[0];
          let minDist = Infinity;
          
          for (const p of baseData) {
            const dist = Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2);
            if (dist < minDist) {
              minDist = dist;
              closest = p;
            }
          }
          
          return closest.label;
        }
      );
      
      return {
        x: point.x,
        y: point.y,
        label: predictedLabel
      };
    }
    
    return point;
  });
}

/**
 * Create a masked visualization dataset for display
 * Shows where the blind spot is while indicating masked regions
 */
export function createVisualizationDataset(
  config: BlindSpotDatasetConfig,
  numSamples: number,
  noiseLevel: number
): Example2D[] {
  const baseData = config.baseDatasetGenerator(numSamples, noiseLevel);
  
  return baseData.map(point => {
    // Mark points in blind spot with a neutral value for visualization
    if (config.blindSpot.isInBlindSpot(point.x, point.y)) {
      return {
        x: point.x,
        y: point.y,
        label: 0  // Neutral value in visualization
      };
    }
    return point;
  });
}

/**
 * Generate statistics about how the blind spot affects the dataset
 */
export function getBlindSpotDatasetStats(
  config: BlindSpotDatasetConfig,
  dataset: Example2D[]
): {
  totalPoints: number;
  maskedPoints: number;
  percentMasked: number;
  avgLabelMagnitudeUnmasked: number;
  avgLabelMagnitudeMasked: number;
} {
  let maskedCount = 0;
  let unmaskedCount = 0;
  let maskedLabelSum = 0;
  let unmaskedLabelSum = 0;

  for (const point of dataset) {
    if (config.blindSpot.isInBlindSpot(point.x, point.y)) {
      maskedCount++;
      maskedLabelSum += Math.abs(point.label);
    } else {
      unmaskedCount++;
      unmaskedLabelSum += Math.abs(point.label);
    }
  }

  return {
    totalPoints: dataset.length,
    maskedPoints: maskedCount,
    percentMasked: (maskedCount / dataset.length) * 100,
    avgLabelMagnitudeUnmasked: unmaskedCount > 0 ? unmaskedLabelSum / unmaskedCount : 0,
    avgLabelMagnitudeMasked: maskedCount > 0 ? maskedLabelSum / maskedCount : 0
  };
}

/**
 * Apply a gradient mask to blind spot (harder masking at center, softer at edges)
 */
export function applyGradientBlindSpotMask(
  config: BlindSpotDatasetConfig,
  numSamples: number,
  noiseLevel: number
): Example2D[] {
  const baseData = config.baseDatasetGenerator(numSamples, noiseLevel);
  
  return baseData.map(point => {
    const maskStrength = config.blindSpot.getBlindSpotMask(point.x, point.y);
    
    // Apply gradient masking: strong masking at center, weak at edges
    return {
      x: point.x,
      y: point.y,
      label: point.label * (1 - maskStrength)
    };
  });
}
