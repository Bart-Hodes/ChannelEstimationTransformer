
%% Clear workspace

clear;
close all;
load('Temp/CDL-B_Channel.mat')

rng(0);


%% DFT codebook

fc = 28e9; % carrier frequency

% Antenna config
Ntv = 1;
Nth = 64;
Nt = Ntv*Nth;
Nrv = 1;
Nrh = 2;
Nr = Nrv*Nrh;
eleSpacing = 0.5; % element spacing, normalized by wavelength

dBdown = 30; % dB
taperz = chebwin(Ntv,dBdown);
tapery = chebwin(Nth,dBdown);
tap = taperz*tapery.'; % Multiply vector tapers to get 8-by-8 taper values

[beamTx,beamAngleTx,beamAngleElTx,beamAngleAzTx,beamElTx,beamAzTx] = getDFTCodebook(Ntv,Nth,eleSpacing,eleSpacing);

% beamTx =  getDFTCodebookCAP(Nth);
antenna = phased.ShortDipoleAntennaElement( ...
    'FrequencyRange', [1e9 100e9]);

arrayTx = phased.ULA('Element', antenna, 'NumElements', Nth, 'ElementSpacing', 0.5 * physconst('LightSpeed') / fc, 'Taper', tap);

% arrayTx = phased.URA('Size',[Ntv Nth],'ElementSpacing',[0.5*physconst('LightSpeed')/fc 0.5*physconst('LightSpeed')/fc],'Taper',tap);
% 
% 
c = physconst('LightSpeed');
for n = [2,29,42,55]

    pattern(arrayTx,fc,[-90:90],0,'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamTx(:,n));hold on;
end

% Check a certain beam (index:18) at transmitter

% % Check beam patterns in azimuth using linear array (transmitter)


%% Perform beam training
[batchSize, Sc, Slot, Rx, Tx] = size(Channel);

H_channel = zeros(batchSize, Sc,Slot,4,Rx);
for batch = 1:batchSize
    for subbatch = 1:Sc
        for SRS = 1:Slot
            Channel_batch = squeeze(Channel(batch,subbatch,:,:,:));
        
            H = squeeze(Channel_batch(SRS,:,:));
            % H = permute(H,[2,1]);
    %         
    %         H= reshape(squeeze(Temp_channel(SRS,:,:,:)),1,64,2)
    %         H = permute(H,[2,3,1]);
    %         
            % Beam sweeping

            for tb = 1:Nt % search all beam pairs
                f = beamTx(:,tb); % Nt x 1
                power(tb) = sum(abs(H*f).^2); % sum over all RF chains
            end

  
            % Order beam pairs in descending order of receive power
            beamTable = zeros(Nt,1);
            tbIdxVec = 1:Nt; % transmit beam index
            
            for bp = 1:(Nt)
                [tB] = find(power == max(max(power)));
                power(tB(1)) = -Inf;
                beamTable(bp,1) = tbIdxVec(tB(1));
            end
            
            beamSelected = [beamTable(1); beamTable(2); beamTable(3); beamTable(4)]; 
            beamSelected_debug(SRS,:) = [beamTable(1); beamTable(2); beamTable(3); beamTable(4)]; 
            %     hold on
            %     figure();pattern(arrayTx,fc,[-180:180],[-89:90],'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamTx(:,beamTable(1)));
            %     pattern(arrayTx,fc,[-180:180],[-89:90],'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamTx(:,beamTable(2)));
            %     pattern(arrayTx,fc,[-180:180],[-89:90],'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamTx(:,beamTable(3)));
            %     pattern(arrayTx,fc,[-180:180],[-89:90],'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamTx(:,beamTable(4)));
            %     
            %     % figure();pattern(arrayTx,fc,[-180:180],[-89:90],'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamRx(:,beamTable(1,2)));
            %     % pattern(arrayTx,fc,[-180:180],[-89:90],'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamRx(:,beamTable(2,2)));
            %     
            %     array = phased.ULA('NumElements',Nth,'ElementSpacing',0.5*physconst('LightSpeed')/fc);
            %     figure();
            %     
            %     pattern(array,fc,[-180:180],0,'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','polar','Type','powerdb','Weights',beamAzTx(:,[beamTable(1),beamTable(2),beamTable(3),beamTable(4)]));hold on;
        end 
    end
        
    [counts, edges] = histcounts(beamSelected_debug, 'BinMethod', 'integers');
    [sortedCounts, idx] = sort(counts, 'descend');
    beamSelected = idx(1:4);

    H_channel_batch = zeros(subbatch,Slot,4,Rx);
    
    for subbatch = 1:Sc
        for SRS = 1:Slot
            for beam = 1:4
                f = beamTx(:,beamSelected(beam));
                temp = squeeze(Channel(batch,subbatch,SRS,:,:));
                H_channel_batch(subbatch,SRS,beam,:) = temp*f;
            end
        end
    end 
    batch
    H_channel(batch,:,:,:,:) = normalize(H_channel_batch);
end

%%
for i = 1:4
    subplot(2,2,i)
%     plot(real(H_channel_batch(:,i,1,1)))
    hold on
    size(H_channel)
    plot(squeeze(real(H_channel(1,1,:,i,1))))
end
save("Temp/RF_Channel.mat","H_channel")