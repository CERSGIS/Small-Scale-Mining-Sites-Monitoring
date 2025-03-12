//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================

// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")
// Map.addLayer(aoi, {}, "Ghana", false);

// Set Parameters
var artSurf = artSurf20;
var notArtSurf = notArtSurf20 ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite_2023_V3');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water_2023_V2').select('classification');
var mangClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mangrove_2023_V2').select('classification');
var wetlandClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Wetland_2023_V2').select('classification');
var miningClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mining_2023_V2').select('classification');


Map.setOptions('satellite');

//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

// mask out vegetated areas
var vegetationMask = composite.select('NDVI').lt(0.65).clip(aoi);
Map.addLayer(vegetationMask.selfMask(), {palette:'red'}, 'Vegetation Mask', false);

//Mask of previously classified pixels
var base = ee.Image.constant(0);
var classMask = waterClass.unmask(base).or(mangClass.unmask()).or(wetlandClass.unmask()).or(miningClass.unmask()).not().clip(aoi);
Map.addLayer(classMask, {palette:['white','blue']}, 'Class Mask', false);

// Mask the composite we are going to use for classification
var compositeMasked = composite.updateMask(classMask).updateMask(vegetationMask);
Map.addLayer(compositeMasked, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Classification');
Map.addLayer(compositeMasked, {bands: ['B7', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Artificial Surfaces', false);
Map.addLayer(compositeMasked, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.2, gamma:1}, 'True Color Composite', false);

//=======================================================================================
//STEP 3: Classify Landsat Image 
//=======================================================================================

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B4','B5','B6','B7','BI','NDVI', 'NDMoI', 'TCB','SR65', 'GCVI', 'HH', 'AVE'];

// Merge the feature collections into a single FeatureCollection.
var sites = artSurf.merge(notArtSurf);

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
  'blue'  // Artificial Surface
];

Map.addLayer (classifiedImage, {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

// =======================================================================================
// STEP 4: Refine Classification
// =======================================================================================

var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(3,'pixels') //define neighborhood
});

Map.addLayer(filterImage,{min:0,max:1,palette: paletteMAP},'Filtered Classification', false);

// Remove Noise
var finalImage = filterImage.select(['classification_mode']).eq(1).selfMask().connectedPixelCount().gte(40).rename('classification');

Map.addLayer(finalImage, {palette: '#1687a7'}, 'Final Classification', true);

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
// Map.addLayer (probImage, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability', false);

// Mask Probability to Artificial Surfaces Class
var probArtSurf = probImage.updateMask(finalImage);
Map.addLayer (probArtSurf, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Artificial Surface Probability', false);

// Add probability layer to export
var exportImage = finalImage.addBands(probArtSurf.rename('probability'));
print('Final Export',exportImage);


//=======================================================================================
//STEP 6: Export Classification
//=======================================================================================

//Export the classification(s)
Export.image.toAsset({
  image: exportImage,
  description: 'Ghana_ArtificialSurfaces',
  scale: 30,
  region: aoi,
  maxPixels:1e13
});

