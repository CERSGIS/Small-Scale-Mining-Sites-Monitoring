//## CREATE A FUNCTION TO CALCULATE SPECTRAL INDICES
// Vegetation Indices
var addIndicesL8 = function(img) {
  // NDVI (Normalized Difference Vegetation Index)
  var ndvi = img.normalizedDifference(['B5','B4']).rename('NDVI');
  // NDMI (Normalized Difference Mangrove Index - Shi et al 2016 )
  var ndmi = img.normalizedDifference(['B7','B3']).rename('NDMI');
  // NDMoI (Normalized Difference Moisture Index)
  var ndmoi = img.normalizedDifference(['B5','B6']).rename('NDMoI');
  // MNDWI (Modified Normalized Difference Water Index - Hanqiu Xu, 2006)
  var mndwi = img.normalizedDifference(['B3','B6']).rename('MNDWI');
  // SR65
  var sr65 = img.select('B6').divide(img.select('B5')).rename('SR65');
  // BI - Baresoil Index
  var bare = img.normalizedDifference(['B6','B7']).rename('BI');
  // GCVI
  var gcvi = img.expression('(NIR/GREEN)-1',{
    'NIR':img.select('B5'),
    'GREEN':img.select('B3')
  }).rename('GCVI');
  //EVI
  var evi = img.expression(
  '2.5 * ((NIR-RED) / (NIR + 6 * RED - 7.5* BLUE +1))', {
    'NIR':img.select('B5'),
    'RED':img.select('B4'),
    'BLUE':img.select('B2')
  }).rename('EVI');
  //NIRv
    var nirv = ee.Image(img).expression(
    '0.08 - ((b("B5") - b("B4")) / (b("B5") + b("B4"))) * b("B4")').rename('NIRv')
  //MSAVI
  var msavi = img.expression(
  '(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - RED)) ) / 2', 
  {
    'NIR': img.select('B5'), 
    'RED': img.select('B4')
  }
  
).rename('MSAVI');
    return img  // add each spectral index to each Landsat scene
    .addBands(ndvi) 
    .addBands(ndmi)
    .addBands(ndmoi)
    .addBands(mndwi)
    .addBands(nirv)
    .addBands(sr65)
    .addBands(evi)
    .addBands(msavi)
    .addBands(gcvi)
    .addBands(bare);
};

// Tasseled Cap Transformations
var tcTransform = function(img){ 
  var b = ee.Image(img).select(["B2", "B3", "B4", "B5", "B6", "B7"]); // select the image bands, bands = B,G,R,NIR,SWIR1,SWIR2
  var brt_coeffs = ee.Image.constant([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]); // set brt coeffs - make an image object from a list of values - each of list element represents a band
  var grn_coeffs = ee.Image.constant([-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446]); // set grn coeffs - make an image object from a list of values - each of list element represents a band
  var wet_coeffs = ee.Image.constant([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]); // set wet coeffs - make an image object from a list of values - each of list element represents a band
  
  var sum = ee.Reducer.sum(); // create a sum reducer to be applyed in the next steps of summing the TC-coef-weighted bands
  var brightness = b.multiply(brt_coeffs).reduce(sum); // multiply the image bands by the brt coef and then sum the bands
  var greenness = b.multiply(grn_coeffs).reduce(sum); // multiply the image bands by the grn coef and then sum the bands
  var wetness = b.multiply(wet_coeffs).reduce(sum); // multiply the image bands by the wet coef and then sum the bands
  var tc = brightness.addBands(greenness)
                    .addBands(wetness)
                    .select([0,1,2], ['TCB','TCG','TCW']); //stack TCG and TCW behind TCB with .addBands, use select() to name the bands
  return img.addBands(tc);
};
// Spatial parameters
var aoi = ee.FeatureCollection('projects/ee-boatennana200/assets/Southern_Ghana_Dissolve')
var version = 'V3'
// Cloud Masking
var maskL8sr = function (image) {
  var cloudShadowBitMask = 1 << 3;
  // var cloudsBitMask = 1 << 2;
  var adjacent = 1 << 1
  var qa = image.select('Fmask');
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
      // .and(qa.bitwiseAnd(cloudsBitMask).eq(0))
      .and(qa.bitwiseAnd(adjacent).eq(0));
  return image.updateMask(mask)
      .select("B[0-9]*")
      .copyProperties(image, ["system:time_start"]);
};


function calcNDFI(image) {
  /* Do spectral unmixing */
  var BANDS = ['B2','B3','B4','B5','B6','B7']
  var gv = [.0500, .0900, .0400, .6100, .3000, .1000]
  var shade = [0, 0, 0, 0, 0, 0]
  var npv = [.1400, .1700, .2200, .3000, .5500, .3000]
  var soil = [.2000, .3000, .3400, .5800, .6000, .5800]
  var cloud = [.9000, .9600, .8000, .7800, .7200, .6500]
  var cf = .3 // Not parameterized
  var cfThreshold = ee.Image.constant(cf)
  var unmixImage = ee.Image(image).select(BANDS).unmix([gv, shade, npv, soil, cloud], true,true)
                  .rename(['band_0', 'band_1', 'band_2','band_3','band_4'])
  var newImage = ee.Image(image).addBands(unmixImage)
  var mask = newImage.select('band_4').lt(cfThreshold)
  var ndfi = ee.Image(unmixImage).expression(
    '((GV / (1 - SHADE)) - (NPV + SOIL)) / ((GV / (1 - SHADE)) + NPV + SOIL)', {
      'GV': ee.Image(unmixImage).select('band_0'),
      'SHADE': ee.Image(unmixImage).select('band_1'),
      'NPV': ee.Image(unmixImage).select('band_2'),
      'SOIL': ee.Image(unmixImage).select('band_3')
    })
    
  return ee.Image(newImage)
        .addBands(ee.Image(ndfi).rename(['NDFI']))
        .select(['band_0','band_1','band_2','band_3','NDFI',])
        .rename(['GV','Shade','NPV','Soil','NDFI'])
        .updateMask(mask)
        .addBands(image)
  }

var collection = ee.ImageCollection("NASA/HLS/HLSL30/v002")
                    .filter(ee.Filter.date('2023-01-01', '2024-06-30'))
                    .filterBounds(aoi)
                    .map(maskL8sr)
                    .map(addIndicesL8) // add indices
                    .map(tcTransform)  // add Tasskled Cap 
                    .map(calcNDFI)
                    .median()
                    .clip(aoi)

var visParams = {
  bands: ['B4', 'B3', 'B2'],
  min:0.01,
  max:0.18,
};

Map.addLayer(collection, {min: -1, max: 1}, 'SentLand');

// Add SAR bands
var dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
                  .filter(ee.Filter.eq('system:index', '2023'))
                  .first();
// print(dataset);

/*Copyright (c) 2021 SERVIR-Mekong
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of the data and associated documentation files, to deal in the data
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies, and to permit persons
to whom the data is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 
THE DATA IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

// Algorithm adapted from https://groups.google.com/g/google-earth-engine-developers/c/ExepnAmP-hQ/m/7e5DnjXXAQAJ

// code pulled from https://mygeoblog.com/2021/01/21/sentinel-1-speckle-filter-refined-lee/

// Speckle Filter 
var RefinedLee = function(img) {
  // img must be in natural units, i.e. not in dB!
  // Set up 3x3 kernels 
  var weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
  var kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, false);

  var mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
  var variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);

  // Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
  var sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);

  var sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, false);

  // Calculate mean and variance for the sampled windows and store as 9 bands
  var sample_mean = mean3.neighborhoodToBands(sample_kernel); 
  var sample_var = variance3.neighborhoodToBands(sample_kernel);

  // Determine the 4 gradients for the sampled windows
  var gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
  gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
  gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
  gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());

  // And find the maximum gradient amongst gradient bands
  var max_gradient = gradients.reduce(ee.Reducer.max());

  // Create a mask for band pixels that are the maximum gradient
  var gradmask = gradients.eq(max_gradient);

  // duplicate gradmask bands: each gradient represents 2 directions
  gradmask = gradmask.addBands(gradmask);

  // Determine the 8 directions
  var directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
  directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
  directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
  directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
  // The next 4 are the not() of the previous 4
  directions = directions.addBands(directions.select(0).not().multiply(5));
  directions = directions.addBands(directions.select(1).not().multiply(6));
  directions = directions.addBands(directions.select(2).not().multiply(7));
  directions = directions.addBands(directions.select(3).not().multiply(8));

  // Mask all values that are not 1-8
  directions = directions.updateMask(gradmask);

  // "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
  directions = directions.reduce(ee.Reducer.sum());  

  //var pal = ['ffffff','ff0000','ffff00', '00ff00', '00ffff', '0000ff', 'ff00ff', '000000'];
  //Map.addLayer(directions.reduce(ee.Reducer.sum()), {min:1, max:8, palette: pal}, 'Directions', false);

  var sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));

  // Calculate localNoiseVariance
  var sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);

  // Set up the 7*7 kernels for directional statistics
  var rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));

  var diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], 
    [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);

  var rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, false);
  var diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, false);

  // Create stacks for mean and variance using the original kernels. Mask with relevant direction.
  var dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
  var dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));

  dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
  dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));

  // and add the bands for rotated kernels
  for (var i=1; i<4; i++) {
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
  }

  // "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
  dir_mean = dir_mean.reduce(ee.Reducer.sum());
  dir_var = dir_var.reduce(ee.Reducer.sum());

  // A finally generate the filtered value
  var varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0));

  var b = varX.divide(dir_var);

  var result = dir_mean.add(b.multiply(img.subtract(dir_mean)));
  return(result.arrayFlatten([['sum']]));
//return(result);
};

// Convert DN values to decibels
var dBconvert = function(image){
  return image.pow(2).log10().multiply(10).subtract(83);
};

// Fill in any masked pixels

var fillmask = function(img){
  // add quality band
  var quality = img.gte(-9999).remap([1],[2]).rename(['quality']).byte();
  var img1 = img.addBands(quality);
  
  // Fill in any remaining values through neighborhood reducer
  var reduce = img.reduceNeighborhood({reducer: ee.Reducer.median(), kernel: ee.Kernel.square(4), skipMasked: false}).clip(aoi) ;
  var quality2 = reduce.gte(-9999).rename(['quality']).byte();
  var img2 = reduce.addBands(quality2).rename(img1.bandNames());

  // Create filled in composite
  var filled = ee.ImageCollection([img1, img2]).qualityMosaic('quality').select(0);
  return filled;
};


// Apply Functions to dataset
var filteredHH = RefinedLee(dataset.select(['HH']));
var filledHH = fillmask(filteredHH);
var dbHH = dBconvert(filledHH);
// print(dbHH);
// Map.addLayer(dbHH.clip(aoi), {min: -25 , max: 0}, 'HH', false);

var filteredHV = RefinedLee(dataset.select(['HV']));
var filledHV = fillmask(filteredHV);
var dbHV = dBconvert(filledHV);
// Map.addLayer(dbHV.clip(aoi), {min: -25 , max: 0}, 'HV');


var correctedSAR = dbHH.rename('HH').addBands(dbHV.rename('HV'));
// print(correctedSAR);

// Radar Indices
var addIndicesSAR = function(img) {
  // Average (AVE)
  var ave = img.select('HH').add(img.select('HV')).divide(2).rename('AVE');
  // Difference (DIF)
  var dif = img.select('HH').subtract(img.select('HV')).rename('DIF');
  // Ratio 1 (RAT1)
  var rat1 = img.select('HH').divide(img.select('HV')).rename('RAT1'); 
  // // Ratio 2 (RAT2)
  // var rat2 = img.select('HV').divide(img.select('HH')).rename('RAT2');
  // // Normalized Difference Index (NDI)
  // var ndi = img.normalizedDifference(['HH','HV']).rename('NDI');
  // // NL Index (NLI)
  // var nli = img.select('HH').multiply(img.select('HV')).divide(img.select('HH').add(img.select('HV'))).rename('NLI');
  
    return img  // add each spectral index to each Landsat scene
    .addBands(ave) 
    .addBands(dif)
    .addBands(rat1);
    // .addBands(rat2)
    // .addBands(ndi)
    // .addBands(nli);
};

var finalSAR = addIndicesSAR(correctedSAR);
// print(finalSAR);
// Map.addLayer(finalSAR.clip(aoi), {bands: ['HH'], min: -25, max: 0}, 'Final SAR Bands', false);

var finalExport = collection.addBands(finalSAR);
// print('Final Composite', finalExport);

//=======================================================================================
//STEP 6: Export Landsat Image 
//=======================================================================================

//Export the classification(s)
Export.image.toAsset({
  image: finalExport,
  description: 'Ghana_Composite',
  scale: 30,
  region: aoi,
  maxPixels: 1e13
}); // pyramid policy: sample

