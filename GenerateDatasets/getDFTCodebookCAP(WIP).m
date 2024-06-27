function [beam] = getDFTCodebookCAP(numEleH)

spacingH = 0.5
fc = 28e9
c = physconst('LightSpeed');

% Resolved azimuth virtual angles
spSampleAz = (2/numEleH)*((1:numEleH)-(numEleH+1)/2)*spacingH; % spatial sampling
a = asind(spSampleAz/spacingH);
b = flip(-a(1:end-1)); % symmetrical angles
beamAngleAz = unique([a b],'stable'); % all beam angles
beamAz = 1/sqrt(numEleH)*exp(1j*2*pi*((0:numEleH-1)-numEleH/2).'*spSampleAz); 

beam = beamAz;
