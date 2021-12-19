function dPrime = GenerateROC(mu_N, mu_SN, sigma, s, ylim_ub, numFrames, bool_save)
    if nargin < 6, numFrames = 1; bool_save = 0; end
    
    dPrime = NaN(1,length(mu_N));
    [pHit, pFalseAlarm] = deal(NaN(length(mu_N), length(s)));
    binSize = diff(s(1:2));
    
    cMap_signal = [0.8500, 0.3250, 0.0980; 0.4660, 0.6740, 0.1880];
    cMap_dPrime = colormap(lines); 
    cMap_dPrime = [cMap_dPrime(1,:); cMap_dPrime(3,:); cMap_dPrime(4,:);...
        cMap_dPrime(6,:); cMap_dPrime(7,:)];
    
    %initialize a movie file
    if bool_save == 1
        mov(1:length(mu_N)*length(s)*numFrames) = struct('cdata',[],'colormap',[]);
        v = VideoWriter('SDT3.avi'); open(v);
    end
    figure
    for j = 1:length(mu_N)
        dPrime(j) = (mu_SN(j) - mu_N(j))/sqrt(0.5*2*sigma);
        probN  = normpdf(s,  mu_N(j), sigma)*binSize;
        probSN = normpdf(s, mu_SN(j), sigma)*binSize;
        for i = 1:length(s)
            for k = 1:numFrames
                pFalseAlarm(j,i) = sum(probN(i:end));
                pHit(j,i) = sum(probSN(i:end));

                subplot(1,2,1)
                fill([s(i:end), fliplr(s(i:end))], [probN(i:end), zeros(1,length(s)-i+1)], ...
                    cMap_signal(1,:),'FaceAlpha', 0.5); hold on;
                fill([s(i:end), fliplr(s(i:end))], [probSN(i:end), zeros(1,length(s)-i+1)], ...
                    cMap_signal(2,:),'FaceAlpha', 0.5);
                plot(s, probN, 'Color', cMap_signal(1,:), 'LineWidth',4); hold on;
                plot(s, probSN, 'Color',cMap_signal(2,:), 'LineWidth',4); hold on;
                plot([s(i), s(i)], [0, ylim_ub], 'Color',cMap_dPrime(j,:), 'LineWidth',4); hold on;
                text(mu_N(j), 0.08, 'N', 'fontSize',15, 'Color', cMap_signal(1,:)); hold on;
                text(mu_SN(j), 0.08, 'N + S', 'fontSize',15, 'Color', cMap_signal(2,:));
                hold off; box off
                ylim([0, ylim_ub]); xlim([s(1), s(end)]); xticks([]); yticks([]);
                xlabel('Internal response');
                title(sprintf(['d'' =  ', num2str(round(dPrime(j),2)), '\n P(Hit) = ',...
                    num2str(round(pHit(j,i),2)),', P(False Alarm) = ',...
                    num2str(round(pFalseAlarm(j,i),2))]));
                set(gca, 'FontSize', 15);

                subplot(1,2,2)
                plot([0,1],[0,1], 'k--', 'LineWidth',2); hold on
                plot(pFalseAlarm(j,1:i), pHit(j,1:i),'.-', 'LineWidth', 3, 'MarkerSize',...
                    13, 'Color', cMap_dPrime(j,:)); box off
                xlim([0,1]); ylim([0,1]); xlabel('P(False Alarm)');
                ylabel('P(Hit)'); xticks([0, 0.5, 1]); yticks([0, 0.5, 1]);
                axis square
                set(gca, 'FontSize', 15); set(gca,'color',[0.9,0.9,0.9]);
                set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.8, 0.5]);
                pause(0.02)
                if bool_save == 1
                    mov(round((j-1)*length(s)*numFrames + (i-1)*numFrames + k)) = getframe(gcf);
                end
            end
        end
    end
    if bool_save == 1; writeVideo(v,mov); close(v); end
end