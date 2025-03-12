//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================

// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")

// Map.addLayer(aoi, {}, "Ghana", false);

// Set Parameters - 
var water = water;
var notWater = notwater ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');

Map.setOptions('satellite');
 
//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

//Visualize
Map.addLayer(composite, {bands: ['TCW']}, 'TCW', false);
Map.addLayer(composite, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, "Composite", false);

// Create a mask to help isolate water pixels

var TCWMask = composite.select(['TCW']).gt(-0.155); 
Map.addLayer(TCWMask.selfMask(), {palette:'blue'}, 'WaterMask', false);

// Mask the composite we are going to use for classification
var compositeMasked = composite.updateMask(TCWMask); // To isolate Just water bodies.
Map.addLayer(compositeMasked, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Classification');


//=======================================================================================
//STEP 3: Classify Landsat Image 
//=======================================================================================

// Composite Bands for Reference:
// B1 - B7, B10, B11 (Landsat)
// dry B3-7 Dry Season Landsat
// wet B3-7 Wet Season Landsat
// NDVI (Normalized Difference Vegetation Index)
// NBR (Normalized Burn Index)
// NDMI (Normalized Difference Mangrove Index - Shi et al 2016 )
// NDMoI (Normalized Difference Moisture Index)
// MNDWI (Modified Normalized Difference Water Index - Hanqiu Xu, 2006)
// SR (Simple Ratio Bands 5 and 4)
// SR65 (Simple Ratio Bands 6 and 5)
// BI - Baresoil Index
// GCVI - Green Chlorophyll Vegetation Index
// EVI - Enhanced Vegetation Index
// MSAVI - Modified Soil-Adjusted Vegetation Index
// TCB - Tassled Cap Brightness
// TCG - Tassled Cap Greeness
// TCW - Tassled Cap Wetness
// TCA - Tassled Cap Angle
// HH and HV - SAR 
// AVE - Average SAR
// DIF - Difference SAR
// RAT1 - Ratio 1 SAR
// RAT2 - Ratio 2 SAR
// NDI - Normalized Difference Index SAR
// NLI - NL Index SAR


// //Test out bands for classification
// Map.addLayer(compositeMasked, {bands: ['DIF'], min:-1, max:1, palette: ['red', 'orange','white', 'green', 'blue']}, "Test Index", false);
// Map.addLayer(compositeMasked, {bands: ['dryB5', 'dryB6', 'dryB4'], min: 0, max: 0.25, gamma:0.7}, 'Dry Classification', false);
// print(compositeMasked.bandNames());

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B4','B5','B6','B7','NDVI','TCW', 'MNDWI', 'HV'];

// Merge the feature collections into a single FeatureCollection.
var sites = water.merge(notwater);

// Train Classifier
var training = compositeMasked.select(bands).sampleRegions({
  collection: sites,
  properties: ['class'],
  scale: 30,
});

var trainedClassifier = ee.Classifier.smileRandomForest(100).train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var classifiedImage = compositeMasked.select(bands).classify(trainedClassifier);

// Create palette
var paletteMAP = [
  'black',  // Other
  'blue'  // Water
];

// Display final layer
Map.addLayer (classifiedImage, {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

//=======================================================================================
//STEP 4: Refine Classification
//=======================================================================================

// Image Filter
var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(2,'pixels') //define neighborhood
});
Map.addLayer(filterImage.eq(1).selfMask(),{palette: '#1687a7'},'Filtered Classification', false);
print(filterImage)
// Prepare image
var finalImage = filterImage.select(['classification_mode']).eq(1).selfMask()
                            .connectedPixelCount().gte(20).selfMask().rename('classification');
Map.addLayer(finalImage, {palette: '#1687a7'}, 'Final Classification');

//=======================================================================================
//STEP 5: Get Probability of Classification
//=======================================================================================

var trainedClassifier = ee.Classifier.smileRandomForest(100).setOutputMode('PROBABILITY').train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var probImage = compositeMasked.select(bands).classify(trainedClassifier);
Map.addLayer (probImage, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability', false);

// Mask Probability to Water Class
var probWater = probImage.updateMask(finalImage);
// Map.addLayer (probWater, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Water Probability');

// Add probability layer to export
var exportImage = finalImage.addBands(probWater.rename('probability'));
print('Final Export',exportImage);

//=======================================================================================
//STEP 6: Export Classification
//=======================================================================================

//Export the classification(s)
Export.image.toAsset({
  image: exportImage,
  description: 'Ghana_Water',
  scale: 30,
  region: aoi,
  maxPixels:1e13
}); //pyramid policy: sample