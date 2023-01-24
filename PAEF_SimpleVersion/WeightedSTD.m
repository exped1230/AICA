%该函数计算加权的标准差，为加权的方差开根号的值
%输入值value为要计算标准差的数据，weight为权重
function STD=WeightedSTD(value,weight)

n=length(value);
if nargin==1
    weight=ones(n,1);
elseif nargin~=2
    error('输入必须为一个向量或者一个向量与对应的权重向量');
end
%求加权平均值
AveValue=dot(value,weight)/sum(weight);

Variance=0;
for i=1:n
    Variance=Variance+weight(i)*(value(i)-AveValue)^2;
end

STD=sqrt(Variance/sum(weight));

end