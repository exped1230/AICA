%�ú�������Contrast of hue
function Contrast=ContrastOfHue(hue)

n=length(hue);
Contrast=0;
for i=1:n-1
    for j=i+1:n
        HueDiff=abs(hue(i)-hue(j));
        Contrast=Contrast+min(HueDiff,1-HueDiff);
    end
end
%�ܸ���Ϊn*(n-1)/2��
Contrast=Contrast/(n*(n-1)/2);