%¸Ãº¯Êý¼ÆËãContrast of brightness
function Contrast=ContrastOfBrightness(value,weight)

Contrast=zeros(1,5);
BrightnessFuzzy=Brightness(value);
for i=1:5
    Contrast(i)=WeightedSTD(BrightnessFuzzy(:,i),weight);
end

end