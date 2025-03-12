//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================
 
// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")
// Map.addLayer(aoi, {}, "Ghana", false);

// Load Layers
var forest = forest20 ;
var notForest = notForest20 ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water').select('classification');
var mangClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mangrove').select('classification');
var wetlandClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Wetland').select('classification');
var miningClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mining').select('classification');
var artSurfClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ArtificialSurfaces').select('classification');
var cForestClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ClosedForest').select('classification');
var woodyCropsClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_WoodyCrops').select('classification');
Map.setOptions('satellite');

//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

// View Reference Layers
var canopy = ee.ImageCollection('users/potapovpeter/GEDI_V27').mosaic();
Map.addLayer(canopy.clip(aoi), {min:0, max:31, palette: ['black', 'white', 'blue', 'green']}, "Canopy Height", false);

// mask out vegetated areas
var vegetationMask = composite.select('NDVI').gt(0.65).clip(aoi);
Map.addLayer(vegetationMask.selfMask(), {palette:'red'}, 'Vegetation Mask', false);

//Mask of previously classified pixels
var base = ee.Image.constant(0);
var classMask = waterClass.unmask(base).or(mangClass.unmask())
                .or(miningClass.unmask()).or(artSurfClass.unmask())
                .or(cForestClass.unmask()).or(woodyCropsClass.unmask())
                .or(wetlandClass.unmask())
                .not().clip(aoi);
Map.addLayer(classMask, {palette:['white','blue']}, 'Class Mask', false);

// Mask the composite we are going to use for classification
var compositeMasked = composite.updateMask(classMask).updateMask(vegetationMask);
Map.addLayer(compositeMasked, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Classification', true);
Map.addLayer(composite,  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.2, gamma:1}, 'True Color Composite', false);


//=======================================================================================
//STEP 3: Classify Landsat Image 
//=======================================================================================

// //Test out bands for classification
// Map.addLayer(compositeMasked, {bands: ['HH'], min:-1, max:1, palette: ['red', 'orange','white', 'green', 'blue']}, "Test Index", false);
// Map.addLayer(compositeMasked, {bands: ['dryB5', 'dryB6', 'dryB4'], min: 0, max: 0.25, gamma:0.7}, 'Dry Composite', false);

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B4','B5','B6','B7','NDVI', 'NDMoI', 'TCG', 'EVI', 'MSAVI', 'HH', 'RAT1'/*, 'dryB5', 'dryB4'*/];

// Merge the feature collections into a single FeatureCollection.
var sites = forest.merge(notForest);

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
  'blue'  // Crops
];

Map.addLayer (classifiedImage, {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

// =======================================================================================
// STEP 4: Refine Classification
// =======================================================================================

var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(2,'pixels') //define neighborhood
});
Map.addLayer(filterImage,{min:0,max:1,palette: paletteMAP},'Filtered Classification', false);

// Remove Noise
var finalImage = filterImage.select(['classification_mode']).eq(1).selfMask().connectedPixelCount().gte(15).rename('classification');

Map.addLayer(finalImage, {palette: '#1687a7'}, 'Final Classification', true);
// Map.addLayer(image, {palette: 'red'}, 'Classification', true);


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

// Mask Probability to Open Forest Class
var probForest = probImage.updateMask(finalImage);
Map.addLayer (probForest, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability');

// Add probability layer to export
var exportImage = finalImage.addBands(probForest.rename('probability'));
print('Final Export',exportImage);


//=======================================================================================
//STEP 6: Export Classification
//=======================================================================================


//Export the classification(s)
Export.image.toAsset({
  image: exportImage,
  description: 'Ghana_OpenForest',
  scale: 30,
  region: aoi,
  maxPixels:1e13
});
