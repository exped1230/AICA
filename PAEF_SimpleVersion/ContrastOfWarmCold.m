%¸Ãº¯Êý¼ÆËãContrast of warm and cold
function Contrast=ContrastOfWarmCold(hue)

n=length(hue);
WarmAndCold=WarmCold(hue);
ContrastWarmCold=zeros(n*(n-1)/2,1);
for i=1:n-1
    for j=i+1:n
        id=i*(2*n-i-1)/2+j-n;
        ContrastWarmCold(id)=dot(WarmAndCold(i,:),WarmAndCold(j,:))/...
            (norm(WarmAndCold(i,:))*norm(WarmAndCold(j,:)));
    end
end

Contrast=sum(ContrastWarmCold)/(n*(n-1)/2);