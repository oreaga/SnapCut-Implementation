function ColorModels = initializeColorModels(IMG, Mask, MaskOutline, LocalWindows, BoundaryWidth, WindowWidth)
% INITIALIZAECOLORMODELS Initialize color models.  ColorModels is a struct you should define yourself.
%
% Must define a field ColorModels.Confidences: a cell array of the color confidence map for each local window.

labImg = rgb2lab(IMG);
maskedImg = labImg.*Mask;
confidences = cell(length(LocalWindows(:,1)), 1);
probMasks = cell(length(LocalWindows(:,1)), 1);
distMatrices = cell(length(LocalWindows(:,1)), 1);
i = 1;
sigSq = (WindowWidth*WindowWidth/2)^2;
halfWidth = ceil(WindowWidth/2);
gmmOldFg = cell(length(LocalWindows(:,1)), 1);
gmmOldBg = cell(length(LocalWindows(:,1)), 1);
gmmNewFg = cell(length(LocalWindows(:,1)), 1);
gmmNewBg = cell(length(LocalWindows(:,1)), 1);
oldFgPixels = cell(length(LocalWindows(:,1)), 1);
oldBgPixels = cell(length(LocalWindows(:,1)), 1);

for center = LocalWindows'
    imgWindow = labImg(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth, :);
    fgWindow = maskedImg(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth, :);
    outlineWindow = MaskOutline(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth);
    
%     figure(1);
%     imshow(fgWindow);
%     figure(2);
%     imshow(outlineWindow);
    
    distMatrix = bwdist(fgWindow(:,:,1));
    bgMask = distMatrix >= BoundaryWidth;
% %     figure(3);
% %     imshow(bgMask);
%     bgMask = or(bgMask,(fgWindow(:,:,1) ~= 0));
% %     imshow(bgMask);
%     bgMask = bgMask == 0;
% %     imshow(bgMask);
    bgWindow = imgWindow.*bgMask;
%     imshow(bgWindow);
    
    fgMask = fgWindow(:,:,1) ~= 0;
    
    fgD1Pixels = imgWindow(:,:,1);
    fgD1Pixels = fgD1Pixels(fgMask);
    fgD2Pixels = imgWindow(:,:,2);
    fgD2Pixels = fgD2Pixels(fgMask);
    fgD3Pixels = imgWindow(:,:,3);
    fgD3Pixels = fgD3Pixels(fgMask);
    bgD1Pixels = imgWindow(:,:,1);
    bgD1Pixels = bgD1Pixels(bgMask);
    bgD2Pixels = imgWindow(:,:,2);
    bgD2Pixels = bgD2Pixels(bgMask);
    bgD3Pixels = imgWindow(:,:,3);
    bgD3Pixels = bgD3Pixels(bgMask);
    
    fgMeans = [mean(fgD1Pixels);
               mean(fgD2Pixels);
               mean(fgD3Pixels)];
           
    bgMeans = [mean(bgD1Pixels);
               mean(bgD2Pixels);
               mean(bgD3Pixels)];
           
    fgPixels = [fgD1Pixels fgD2Pixels fgD3Pixels];
    bgPixels = [bgD1Pixels bgD2Pixels bgD3Pixels];
    fgPixelDistWeights = distMatrix(fgMask);
    fgPixelDistWeights = exp((fgPixelDistWeights.*fgPixelDistWeights)/sigSq);
    bgPixelDistWeights = distMatrix(bgMask);
    bgPixelDistWeights = exp((bgPixelDistWeights.*bgPixelDistWeights)/sigSq);
    
    fgCov = cov(fgPixels);
    bgCov = cov(bgPixels);
    
    fgGMM = gmdistribution(fgMeans', fgCov);
    bgGMM = gmdistribution(bgMeans', bgCov);
    
    gmmOldFg{i} = fgGMM;
    gmmOldBg{i} = bgGMM;
             
    
    fgPostProbsFgPixels = pdf(fgGMM, fgPixels);
    bgPostProbsFgPixels = pdf(bgGMM, fgPixels);
    fgPostProbsBgPixels = pdf(fgGMM, bgPixels);
    bgPostProbsBgPixels = pdf(bgGMM, bgPixels);
    totalProbsFgPixels = fgPostProbsFgPixels + bgPostProbsFgPixels;
    totalProbsBgPixels = fgPostProbsBgPixels + bgPostProbsBgPixels;
    fgProbsFgPixels = fgPostProbsFgPixels./totalProbsFgPixels;
    fgProbsBgPixels = fgPostProbsBgPixels./totalProbsBgPixels;
    fgPixelConfidencesSum = fgSum(fgProbsFgPixels, fgPixelDistWeights);
    bgPixelConfidencesSum = bgSum(fgProbsBgPixels, bgPixelDistWeights);
    
    totalConfNumSum = fgPixelConfidencesSum + bgPixelConfidencesSum;
    totalWeightSum = sum(fgPixelDistWeights) + sum(bgPixelDistWeights);
    
    fc = 1 - totalConfNumSum/totalWeightSum;
    
    numEls = numel(imgWindow(:,:,1));
    allPixels2D = reshape(imgWindow, numEls, 3);
    
    fgPostProbsAllPixels = pdf(fgGMM, allPixels2D);
    bgPostProbsAllPixels = pdf(bgGMM, allPixels2D);
    fgProbsAllPixels = fgPostProbsAllPixels./(fgPostProbsAllPixels + bgPostProbsAllPixels);
    fgProbsMask = reshape(fgProbsAllPixels, halfWidth*2, halfWidth*2);
    
    confidences{i} = fc;
    probMasks{i} = fgProbsMask;
    distMatrices{i} = distMatrix;
    oldFgPixels{i} = fgPixels;
    oldBgPixels{i} = bgPixels;
    
    
    i = i + 1;
    
    
end

ColorModels = struct;
ColorModels.Confidences = confidences;
ColorModels.FgProbMasks = probMasks;
ColorModels.DistMatrices = distMatrices;
ColorModels.GmmOldFgs = gmmOldFg;
ColorModels.GmmOldBgs = gmmOldBg;
ColorModels.GmmNewFgs = gmmNewFg;
ColorModels.GmmNewBgs = gmmNewBg;
ColorModels.OldFgPixels = oldFgPixels;
ColorModels.OldBgPixels = oldBgPixels;

end

function fgPixelConfidencesSum = fgSum(fgProbsFgPixels, fgPixelDistWeights)
    complementProbs = 1 - fgProbsFgPixels;
    probTimesWeights = complementProbs.*fgPixelDistWeights;
    fgPixelConfidencesSum = sum(probTimesWeights);
end

function bgPixelConfidencesSum = bgSum(fgProbsBgPixels, bgPixelDistWeights)
    complementProbs = fgProbsBgPixels;
    probTimesWeights = complementProbs.*bgPixelDistWeights;
    bgPixelConfidencesSum = sum(probTimesWeights);
end

