%该函数计算Itten Contrast
function IttenContrast=IttenColorContrast(H,S,V,L,colornames)

NumberOfSegments=size(unique(L),1);%求分割块的个数
SegmentsHSV=zeros(NumberOfSegments,3);
SegmentSize=zeros(NumberOfSegments,1);
for i=1:NumberOfSegments
    SegmentI=(L==i-1);
    %求每个块的像素个数或块的大小
    SegmentSize(i)=sum(sum(SegmentI));
    %求每个块的平均HSV
    SegmentsHSV(i,1)=sum(sum(H.*SegmentI))/SegmentSize(i);
    SegmentsHSV(i,2)=sum(sum(S.*SegmentI))/SegmentSize(i);
    SegmentsHSV(i,3)=sum(sum(V.*SegmentI))/SegmentSize(i);
end

% IttenSaturation=Saturation(SegmentsHSV(:,2));
% IttenBrightness=Brightness(SegmentsHSV(:,3));

%ContrastOfHue1(SegmentsHSV(:,1),L)
IttenContrast=cat(2,ContrastOfHue(SegmentsHSV(:,1)),ContrastOfSaturation(SegmentsHSV(:,2),SegmentSize),...
    ContrastOfBrightness(SegmentsHSV(:,3),SegmentSize),ContrastOfWarmCold(SegmentsHSV(:,1)),...
    ContrastOfComplementsAndSimultaneous(SegmentsHSV(:,1)),ContrastOfExtension(colornames));

end