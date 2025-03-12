//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================

// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")
// Map.addLayer(aoi, {}, "Ghana", false);


// Set Parameters - 
var mining = mining20;
var notMining = notMining20 ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water').select('classification');
var mangClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mangrove').select('classification');
var wetlandClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Wetland').select('classification');


Map.setOptions('satellite');

//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

// mask out vegetated areas
var vegetationMask = composite.select('NDVI').lt(0.65).clip(aoi);
Map.addLayer(vegetationMask.selfMask(), {palette:'red'}, 'Vegetation Mask', false);

//Mask of previously classified pixels
var base = ee.Image.constant(0);
var classMask = waterClass.unmask(base).or(mangClass.unmask()).or(wetlandClass.unmask()).not().clip(aoi);
// Map.addLayer(classMask, {palette:['white','blue']}, 'Class Mask', false);

// Mask out northern part of country
var blank = ee.Image.constant(0).clip(aoi);
var regionMask = blank.paint(south, 1);
Map.addLayer(regionMask.clip(aoi), {min:0, max:1, palette:['white','blue']}, 'Region Mask', false);


// Mask the composite we are going to use for classification
var compositeMasked = composite.updateMask(classMask).updateMask(vegetationMask).updateMask(regionMask);
Map.addLayer(compositeMasked, {bands: ["B4","B3","B2"],gamma: 1.3450000000000002,max: 0.1726137710571289,min: 0.012759654846191402,opacity: 1}, 'Composite for Classification');


//=======================================================================================
//STEP 3: Classify Landsat Image 
//=======================================================================================

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B4','B5','B6','B7','BI','NDVI', 'NDMoI', 'TCB', 'SR65', 'AVE', 'HH'];
// var bands = ['TCB', 'B2', 'B11', 'B10', 'BI', 'B5', 'RAT1', 'NIRv', 'TCG', 'DIF']

// Merge the feature collections into a single FeatureCollection.
var sites = mining.merge(notMining);

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
  'blue'  // Mining
];

Map.addLayer (classifiedImage.selfMask(), {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

// =======================================================================================
// STEP 4: Refine Classification
// =======================================================================================

var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(3,'pixels') //define neighborhood
});

Map.addLayer(filterImage,{min:0,max:1,palette: paletteMAP},'Filtered Classification', false);

// Remove Noise
var finalImage = filterImage.select(['classification_mode']).eq(1).selfMask().connectedPixelCount().gte(20).rename('classification');

Map.addLayer(finalImage, {palette: '#1687a7'}, 'Final Classification', true);


//=======================================================================================
//STEP 5: Get Probability of Classification
//=======================================================================================

// Train Classifier
var trainedClassifier = ee.Classifier.smileRandomForest(100).setOutputMode('PROBABILITY').train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var probImage = compositeMasked.select(bands).classify(trainedClassifier);

// Mask Probability to Mining Class
var probMining = probImage.updateMask(finalImage);
Map.addLayer (probMining, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Mining Probability');

// Add probability layer to export
var exportImage = finalImage.addBands(probMining.rename('probability'));
print('Final Export',exportImage);

//=======================================================================================
//STEP 6: Export
//=======================================================================================

//Export the classification(s)
Export.image.toAsset({
  image: exportImage,
  description: 'Ghana_Mining',
  scale: 30,
  region: aoi,
  maxPixels:1e13
});