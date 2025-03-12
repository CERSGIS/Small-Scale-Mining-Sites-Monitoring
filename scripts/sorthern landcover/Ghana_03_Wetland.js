//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================

// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")

// Map.addLayer(aoi, {}, "Ghana", false);

// Set Parameters - 2023
var year = '_2023' ;
var wetland = wetland20.merge(wetland15);
var notWetland = notWetland20.merge(notWetland15) ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water').select('classification');
var mangClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mangrove').select('classification');

Map.setOptions('satellite');

//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

// View Reference Layers
var canopy = ee.ImageCollection('users/potapovpeter/GEDI_V27').mosaic();
Map.addLayer(canopy.clip(aoi), {min:0, max:31, palette: ['black', 'white', 'blue', 'green']}, "Canopy Height", false);

// mask to lower elevation regions
var elevation = ee.Image("USGS/SRTMGL1_003").select('elevation');
var elevationMask = elevation.lt(85).clip(aoi); 
// Map.addLayer(elevationMask.selfMask(), {palette:'blue'}, 'Elevation Mask', false);

//Mask of previously classified pixels
var base = ee.Image.constant(0);
var classMask = waterClass.unmask(base).or(mangClass.unmask()).not().clip(aoi);
Map.addLayer(classMask, {palette:['white','blue']}, 'Class Mask', false);

// Mask the composite we are going to use for classification
var compositeMasked = composite.addBands(elevation).updateMask(classMask).updateMask(elevationMask);
Map.addLayer(compositeMasked, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Classification');


//=======================================================================================
//STEP 3: Classify Landsat Image 
//=======================================================================================

 // //Test out bands for classification
Map.addLayer(compositeMasked, {bands: ['NDVI'], min:0, max:1, palette: ['red', 'orange','white', 'green', 'blue']}, "Test Index", false);

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B3','B4','B5','B6','B7', 'NDMI', 'NDVI', 'TCW', 'MSAVI', 'HH', 'HV','elevation'];

// Merge the feature collections into a single FeatureCollection.
var sites = wetland.merge(notWetland);

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
  'blue'  // Forest
];

Map.addLayer (classifiedImage, {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

// =======================================================================================
// STEP 4: Refine Classification
// =======================================================================================

var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(4,'pixels') //define neighborhood
});

Map.addLayer(filterImage,{min:0,max:1,palette: paletteMAP},'Filtered Classification', false);

// Remove Noise 
var finalImage = filterImage.select(['classification_mode']).eq(1).selfMask().connectedPixelCount().gte(10).rename('classification');

Map.addLayer(finalImage, {palette: '#1687a7'}, 'Final Classification', true);


//=======================================================================================
//STEP 6: Get Probability of Classification
//=======================================================================================

var trainedClassifier = ee.Classifier.smileRandomForest(100).setOutputMode('PROBABILITY').train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var probImage = compositeMasked.select(bands).classify(trainedClassifier);
// Map.addLayer (probImage, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability', false);

// Mask Probability to Wetland Class
var probWetland = probImage.updateMask(finalImage);
Map.addLayer (probWetland, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability', false);

// Add probability layer to export
var exportImage = finalImage.addBands(probWetland.rename('probability'));
print('Final Export',exportImage);

//=======================================================================================
//STEP 5: Export Classification
//=======================================================================================


//Export the classification(s)
Export.image.toAsset({
  image: exportImage,
  description: 'Ghana_Wetland',
  scale: 30,
  region: aoi,
  maxPixels:1e13
});