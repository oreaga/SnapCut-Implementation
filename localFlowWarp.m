function [NewLocalWindows] = localFlowWarp(WarpedPrevFrame, CurrentFrame, LocalWindows, Mask, Width)
% LOCALFLOWWARP Calculate local window movement based on optical flow between frames.

% TODO

opticalFlow = opticalFlowHS;

flow = estimateFlow(opticalFlow, rgb2gray(WarpedPrevFrame));
flow = estimateFlow(opticalFlow, rgb2gray(CurrentFrame));

halfWidth = ceil(Width/2);
NewLocalWindows = LocalWindows;

% imshow(WarpedPrevFrame) 
% hold on
% plot(flow,'DecimationFactor',[5 5],'ScaleFactor',700)
% hold off

for i = 1:length(LocalWindows(:,1))
    center = ceil(LocalWindows(i,:));
    flowWindowX = flow.Vx(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth);
    flowWindowY = flow.Vy(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth);
    maskWindow = Mask(center(2)-halfWidth+1:center(2)+halfWidth, center(1)-halfWidth+1:center(1)+halfWidth);
    
    avgVx = mean(mean(flowWindowX.*maskWindow));
    avgVy = mean(mean(flowWindowY.*maskWindow));
    
    newXValue = ceil(LocalWindows(i,1) + avgVx);
    newYValue = ceil(LocalWindows(i,2) + avgVy);
    NewLocalWindows(i,:) = [newXValue newYValue];
end



end

