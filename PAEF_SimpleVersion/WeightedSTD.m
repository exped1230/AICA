%�ú��������Ȩ�ı�׼�Ϊ��Ȩ�ķ�����ŵ�ֵ
%����ֵvalueΪҪ�����׼������ݣ�weightΪȨ��
function STD=WeightedSTD(value,weight)

n=length(value);
if nargin==1
    weight=ones(n,1);
elseif nargin~=2
    error('�������Ϊһ����������һ���������Ӧ��Ȩ������');
end
%���Ȩƽ��ֵ
AveValue=dot(value,weight)/sum(weight);

Variance=0;
for i=1:n
    Variance=Variance+weight(i)*(value(i)-AveValue)^2;
end

STD=sqrt(Variance/sum(weight));

end