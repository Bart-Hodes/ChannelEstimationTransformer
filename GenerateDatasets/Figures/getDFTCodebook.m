function [beam,beamAngle,beamAngleEl,beamAngleAz,beamEl,beamAz] = getDFTCodebook(numEleV,numEleH,spacingV,spacingH)

% Resolved elevation virtual angles
spSampleEl = (0:numEleV-1)./numEleV; % spatial sampling
a = unique(real(asind(spSampleEl./spacingV)),'stable'); % spatial angles
b = flip(-a(1:end-1)); % symmetrical angles
beamAngleEl = unique([a b],'stable'); % all beam angles
beamEl = 1/sqrt(numEleV)*exp(1j*2*pi*((0:numEleV-1)-numEleV/2).'*spSampleEl); 

% Resolved azimuth virtual angles
spSampleAz = (0:numEleH-1)./numEleH; % spatial sampling
a = unique(real(asind(spSampleAz./spacingH)),'stable'); % spatial angles
b = flip(-a(1:end-1)); % symmetrical angles
beamAngleAz = unique([a b],'stable'); % all beam angles
beamAz = 1/sqrt(numEleH)*exp(1j*2*pi*((0:numEleH-1)-numEleH/2).'*spSampleAz); 

numEle = numEleV*numEleH;
beam = zeros(numEle,numEle);
beamAngle = zeros(2,numEle);
idx = 1;
for nv = 1:numel(beamAngleEl)
    for nh = 1:numel(beamAngleAz)
        
        beam2D = flip(beamEl(:,nv)).*beamAz(:,nh).'; % element-wise
        beam(:,idx) = beam2D(:);
        beamAngle(:,idx) = [beamAngleEl(nv);beamAngleAz(nh)];
        idx = idx+1;
    end
end

