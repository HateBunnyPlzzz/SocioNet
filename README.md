# README 

## Project Summary

This project, named SocioNet, employs advanced AI techniques to estimate poverty at the district level in India, leveraging socio-economic data and satellite imagery. The aim is to provide actionable insights for poverty eradication efforts by accurately predicting poverty levels using a novel data integration and analysis approach.

## Project Components and File Structure

### Available Datasets

After initial data pre-processing, our datasets, such as NFHS-4, are structured as follows:

| Country | Cluster Number | District | Year | ELECTHH | COOKFUEL | TOILETTYPE | DRINKWATER | WEALTHQHH | WEALTHSHH |
| ------- | -------------- | -------- | ---- | ------- | -------- | ---------- | ---------- | --------- | --------- |
| IA6     | 100001         | 585      | 2015 | 1       | 2        | 11         | 13         | 5         | 1.97208   |
| IA6     | 100001         | 585      | 2015 | 1       | 2        | 12         | 11         | 5         | 1.76550   |
| IA6     | 100001         | 585      | 2015 | 1       | 2        | 12         | 11         | 4         | 0.79116   |
| IA6     | 100001         | 585      | 2015 | 1       | 2        | 12         | 11         | 5         | 2.13757   |
| IA6     | 100001         | 585      | 2015 | 1       | 2        | 11         | 12         | 4         | 0.94395   |

### Satellite Imagery Acquisition

Satellite images were downloaded using the Google Earth Engine console with JavaScript for processing and analysis. The script creates 10km x 10km square buffers around district coordinates and fetches median composite images from Sentinel-2 for the year 2015.

#### JavaScript for Google Earth Engine:

This script includes functions for creating bounding boxes around districts, fetching Sentinel-2 composite images, and setting up export tasks for each district image.

```javascript
// Load the FeatureCollection of district coordinates.
var districtCoordinates = ee.FeatureCollection('projects/scenic-lane-400211/assets/top_100_district_coordinates_2015_');

// Function to create a 10km x 10km square buffer around each point (district's coordinates).
function createBoundingBox(feature) {
  var lat = feature.get('LATNUM');
  var lon = feature.get('LONGNUM');
  var point = ee.Geometry.Point([lon, lat]);
  var square = point.buffer(10000 / 2 * Math.sqrt(2)).bounds();
  return feature.setGeometry(square);
}

// Apply the function to all districts to create their bounding boxes.
var districtBoundingBoxes = districtCoordinates.map(createBoundingBox);

// Sentinel-2 composite image for the specified time range.
var sentinel2Composite = ee.ImageCollection('COPERNICUS/S2')
                            .filterDate('2015-01-01', '2015-12-31')
                            .filterBounds(districtBoundingBoxes)
                            .median();

// Function to set up export tasks for each district
function setupExport(districtFeature, index) {
  var districtImage = sentinel2Composite.clip(districtFeature.geometry());

  // Use district ID for naming, ensure it's fetched correctly before this point
  var districtId = districtFeature.get('District').getInfo(); // This should be adjusted

  Export.image.toDrive({
    image: districtImage,
    description: 'DistrictImage_' + districtId,
    folder: 'GEE_District_Images',
    scale: 10,
    region: districtFeature.geometry().bounds(),
    maxPixels: 1e13
  });
}

function visualizeDistrictImage(districtFeature) {
  var districtImage = sentinel2Composite.clip(districtFeature.geometry());
  
  // Generate a unique name for the layer based on the district ID
  var districtId = districtFeature.get('District').getInfo(); // This is synchronous and should be used cautiously
  
  // Add the clipped district image to the map
  Map.addLayer(districtImage, {bands: ['B4', 'B3', 'B2'], max: 3000}, 'District Image ' + districtId);
}

// Manual iteration for batch processing
// Here, you manually update 'batchStart' for subsequent batches
var batchSize = 10; // Define the size of each batch
var batchStart = 0; // Update this manually for each batch

// Select a batch of districts based on 'batchStart' and 'batchSize'
var batchDistricts = districtBoundingBoxes.toList(batchSize, batchStart);

// Manually initiate export for each district in the batch
for (var i = 0; i < batchSize; i++) {
  var districtFeature = ee.Feature(batchDistricts.get(i));
  setupExport(districtFeature, batchStart + i); // Pass the correct index or ID
}
```

## Notebooks and Code Files

### Feature Fusion (`feature_fusion.ipynb`)

This Jupyter notebook performs the crucial step of integrating features extracted from satellite imagery via Vision Transformer (ViT) models with tabular socioeconomic data. The notebook outlines the process of data import, dimensionality reduction via PCA, data merging, and final dataset creation.

**Usage Notes:**
- Ensure the prerequisite datasets are located in the same directory as the notebook.
- Adjust the `n_components` parameter in the PCA step according to your analysis.

### Model Training (`model_training.ipynb`)

This Jupyter notebook outlines the process of training machine learning models to predict district-level wealth indices using socioeconomic data, with and without Vision Transformer (ViT) extracted features. The notebook includes environment setup, data loading, model training without and with ViT features, performance evaluation, and final model ensemble.

**Final Output:**
- Model without ViT features: RMSE = 0.2174350037317734, R2 = 0.9541821543423477
- Model with ViT features: RMSE = 0.21703717511589368, R2 = 0.9543496616411186

**Usage Notes:**
- Ensure the datasets are present in the project directory before running the notebook.
- Review and adjust the `max_epochs`, `patience`, and `eval_metric` parameters as needed.

## Installation and Setup

---

## Environment Setup and Requirements

To run the SocioNet project, you'll need a Python environment with specific libraries installed. Below are the instructions for setting up this environment and the required dependencies.

### Python Version

Ensure you have Python 3.8 or newer installed on your system. You can download Python from [the official website](https://www.python.org/downloads/).

### Required Packages

The project requires PyTorch 1.2 or newer, along with several other libraries for data manipulation, modeling, and computation. You can install all required dependencies using the following commands:

1. **PyTorch**: Follow the [official installation guide](https://pytorch.org/get-started/locally/) to install PyTorch. Ensure you select the correct version based on your system's CUDA compatibility to leverage GPU acceleration (if available).

2. **Additional Dependencies**: Install the remaining required libraries using pip. Here's a list of some essential libraries and their recommended versions:

```bash
pip install pandas>=1.1
pip install numpy>=1.19
pip install scikit-learn>=0.23
pip install xgboost>=1.3
pip install pytorch-tabnet>=3.1
pip install matplotlib>=3.3
pip install seaborn>=0.11
```

### Creating a Virtual Environment (Optional)

It's a good practice to create a virtual environment for your project to avoid conflicts with other projects' dependencies. Use the following commands to create and activate a virtual environment:

```bash
# Install virtualenv if you haven't installed it yet
pip install virtualenv

# Create a virtual environment named 'socionet_env'
virtualenv socionet_env

# Activate the virtual environment
# On Windows
.\socionet_env\Scripts\activate
# On Unix or MacOS
source socionet_env/bin/activate

# Now, you can install the required packages within this environment
```

After setting up the environment and installing the required packages, you can proceed to run the project's notebooks and scripts.


## License

```
Copyright 2024 Shubhanshu Khatana

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
