function [WarpedFrame, WarpedMask, WarpedMaskOutline, WarpedLocalWindows] = calculateGlobalAffine(IMG1,IMG2,Mask,Windows)
% CALCULATEGLOBALAFFINE: finds affine transform between two frames, and applies it to frame1, the mask, and local windows.
    
    gs1 = rgb2gray(IMG1);
    gs2 = rgb2gray(IMG2);

    img1Features = detectHarrisFeatures(gs1);
    img2Features = detectHarrisFeatures(gs2);
    
    [extractedFeatures1, validPts1] = extractFeatures(gs1, img1Features);
    [extractedFeatures2, validPts2] = extractFeatures(gs2, img2Features);
    
    indexPairs = matchFeatures(extractedFeatures1, extractedFeatures2);
    
    matchedPts1 = validPts1(indexPairs(:,1));
    matchedPts2 = validPts2(indexPairs(:,2));
    
%     figure(10);
%     showMatchedFeatures(gs1, gs2, matchedPts1, matchedPts2, 'montage');
    
    [tForm, inlierPts1, inlierPts2] = estimateGeometricTransform(matchedPts1, matchedPts2, 'affine');
    
    imageRef = imref2d(size(IMG1));
    
    WarpedFrame = imwarp(IMG1, tForm, 'OutputView', imageRef);
    
    WarpedMask = imwarp(Mask, tForm, 'OutputView', imageRef);
    
    [WarpedLocalWindowsX, WarpedLocalWindowsY] = transformPointsForward(tForm, Windows(:,1), Windows(:,2));
    WarpedLocalWindows = [WarpedLocalWindowsX WarpedLocalWindowsY];
    
    WarpedMaskOutline = bwperim(WarpedMask);
end

