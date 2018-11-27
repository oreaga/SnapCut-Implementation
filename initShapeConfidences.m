function ShapeConfidences = initShapeConfidences(LocalWindows, ColorModels, WindowWidth, SigmaMin, A, fcutoff, R)
% INITSHAPECONFIDENCES Initialize shape confidences.  ShapeConfidences is a struct you should define yourself.

i = 1;
shapeConfidenceMatrices = cell(length(LocalWindows(:,1)), 1);

while i <= length(LocalWindows(:,1))
   fc = ColorModels.Confidences{i};
   distMatrix = ColorModels.DistMatrices{i};
   
   if fc > fcutoff
       sigma = SigmaMin + A*((fc - fcutoff)^R);
   else
       sigma = SigmaMin;
   end
   
   sigSquared = sigma^2;
   distSquaredMatrix = distMatrix.*distMatrix;
   fgShapeConfidences = 1 - exp(-distSquaredMatrix/sigSquared);
   
   shapeConfidenceMatrices{i} = fgShapeConfidences;
   i = i + 1;
end

ShapeConfidences = struct;

ShapeConfidences.ShapeConfidenceMatrices = shapeConfidenceMatrices;

end
