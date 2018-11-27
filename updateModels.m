function [mask, LocalWindows, ColorModels, ShapeConfidences] = ...
    updateModels(...
        NewLocalWindows, ...
        LocalWindows, ...
        CurrentFrame, ...
        warpedMask, ...
        warpedMaskOutline, ...
        WindowWidth, ...
        ColorModels, ...
        ShapeConfidences, ...
        ProbMaskThreshold, ...
        fcutoff, ...
        SigmaMin, ...
        R, ...
        A ...
    )
% UPDATEMODELS: update shape and color models, and apply the result to generate a new mask.
% Feel free to redefine this as several different functions if you prefer.

halfWidth = ceil(WindowWidth/2);
numEl = (halfWidth*2)^2;
sigSq = (WindowWidth*WindowWidth/2)^2;
LabFrame = rgb2lab(CurrentFrame);

for j = 1:1
    mergedFgProbMapNum = zeros(size(CurrentFrame(:,:,1)));
    mergedFgProbMapDenom = zeros(size(CurrentFrame(:,:,1)));
    for i = 1:length(NewLocalWindows(:,1))
        center = LocalWindows(i,:);
        labWindow = LabFrame(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth, :);
        labPixels = reshape(labWindow, numEl, 3);

        pixelPostProbsHistFg = pdf(ColorModels.GmmOldFgs{i}, labPixels);
        pixelPostProbsHistBg = pdf(ColorModels.GmmOldBgs{i}, labPixels);
        pixelProbsHist = pixelPostProbsHistFg./(pixelPostProbsHistFg + pixelPostProbsHistBg); 
        fgThreshPixelMask = pixelProbsHist > 0.75;
        bgThreshPixelMask = pixelProbsHist < 0.25;
        fgThreshPixels = labPixels(fgThreshPixelMask, :);
        bgThreshPixels = labPixels(bgThreshPixelMask, :);
        fgAllPixels = [fgThreshPixels;
                       ColorModels.OldFgPixels{i}];
        bgAllPixels = [bgThreshPixels;
                       ColorModels.OldBgPixels{i}];

        newFgGmm = gmdistribution(mean(fgAllPixels, 1), cov(fgAllPixels));
        newBgGmm = gmdistribution(mean(bgAllPixels, 1), cov(bgAllPixels));


        newFgPostProbs = pdf(newFgGmm, labPixels);
        newBgPostProbs = pdf(newBgGmm, labPixels);
        newFgProbs = reshape(newFgPostProbs./(newFgPostProbs + newBgPostProbs), halfWidth*2, halfWidth*2);
        oldFgProbs = reshape(pixelProbsHist, halfWidth*2, halfWidth*2);
        newFgMaskPixels = newFgProbs > ProbMaskThreshold;
        oldFgMaskPixels = oldFgProbs > ProbMaskThreshold;
        windowDistMatrix = bwdist(warpedMaskOutline(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth, :));
        windowFgMask = warpedMask(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth, :);


        if (sum(newFgMaskPixels(:)) <= sum(oldFgMaskPixels(:)))
            colorConfidence = computeColorConfidence(windowDistMatrix, windowFgMask, newFgProbs, sigSq);
            ColorModels.Confidences{i} = colorConfidence;
            ColorModels.FgProbMasks{i} = newFgProbs;
            ColorModels.GmmOldFgs{i} = newFgGmm;
            ColorModels.GmmOldBgs{i} = newBgGmm;
        else
            ColorModels.FgProbMasks{i} = oldFgProbs;
        end

       fc = ColorModels.Confidences{i};

       if fc > fcutoff
           sigma = SigmaMin + A*((fc - fcutoff)^R);
       else
           sigma = SigmaMin;
       end

       sigSquared = sigma^2;
       distSquaredMatrix = windowDistMatrix.*windowDistMatrix;
       fgShapeConfidences = 1 - exp(-distSquaredMatrix/sigSquared);
       ShapeConfidences.ShapeConfidenceMatrices{i} = fgShapeConfidences;

       combinedFgProbMap = fgShapeConfidences.*windowFgMask + (1 - fgShapeConfidences).*ColorModels.FgProbMasks{i};
       ColorModels.FgProbMasks{i} = combinedFgProbMap;

       x = repmat(1:halfWidth*2, halfWidth*2, 1);
       y = repmat(transpose(halfWidth*2:-1:1), 1, halfWidth*2);
       c = [halfWidth halfWidth];
       centerDistanceMatrix = sqrt((y - c(1)) .^ 2 + (x - c(2)) .^ 2) + 0.1;
       invCenterDistanceMatrix = 1./centerDistanceMatrix;

       mergedFgProbMapNum(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth) = mergedFgProbMapNum(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth) + combinedFgProbMap.*invCenterDistanceMatrix;

       mergedFgProbMapDenom(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth) = ...
           mergedFgProbMapDenom(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth)...
           + invCenterDistanceMatrix;

    end

    mergedFgProbMap = mergedFgProbMapNum./mergedFgProbMapDenom;
    mask = mergedFgProbMap > ProbMaskThreshold;
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 25);
%     L = superpixels(CurrentFrame, 10000);
%     mask = lazysnapping(CurrentFrame, L, mask, mask ~= 1);
    [maskOutline, LocalWindows] = initLocalWindows(CurrentFrame, mask, length(LocalWindows(:,1)), WindowWidth, false);
    mask = imfill(maskOutline, 'holes');
end

end

function colorConfidence = computeColorConfidence(WindowDistMatrix, WindowFgMask, NewFgProbs, sigSq)
    bgMask = WindowFgMask ~= 1;
    fgProbsFgPixels = NewFgProbs(WindowFgMask);
    fgProbsBgPixels = NewFgProbs(bgMask);
    fgPixelDistWeights = WindowDistMatrix(WindowFgMask);
    fgPixelDistWeights = exp((fgPixelDistWeights.*fgPixelDistWeights)/sigSq);
    bgPixelDistWeights = WindowDistMatrix(bgMask);
    bgPixelDistWeights = exp((bgPixelDistWeights.*bgPixelDistWeights)/sigSq);
    fgPixelConfidencesSum = fgSum(fgProbsFgPixels, fgPixelDistWeights);
    bgPixelConfidencesSum = bgSum(fgProbsBgPixels, bgPixelDistWeights);
    
    totalConfNumSum = fgPixelConfidencesSum + bgPixelConfidencesSum;
    totalWeightSum = sum(fgPixelDistWeights) + sum(bgPixelDistWeights);
    
    colorConfidence = 1 - totalConfNumSum/totalWeightSum;
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

