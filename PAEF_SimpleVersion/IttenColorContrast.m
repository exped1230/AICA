%�ú�������Itten Contrast
function IttenContrast=IttenColorContrast(H,S,V,L,colornames)

NumberOfSegments=size(unique(L),1);%��ָ��ĸ���
SegmentsHSV=zeros(NumberOfSegments,3);
SegmentSize=zeros(NumberOfSegments,1);
for i=1:NumberOfSegments
    SegmentI=(L==i-1);
    %��ÿ��������ظ������Ĵ�С
    SegmentSize(i)=sum(sum(SegmentI));
    %��ÿ�����ƽ��HSV
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