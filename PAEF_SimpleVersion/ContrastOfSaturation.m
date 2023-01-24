%¸Ãº¯Êý¼ÆËãContrast of Saturation
function Contrast=ContrastOfSaturation(color,weight)

Contrast=zeros(1,3);
SaturationFuzzy=Saturation(color);

for i=1:3
    Contrast(i)=WeightedSTD(SaturationFuzzy(:,i),weight);
end

end