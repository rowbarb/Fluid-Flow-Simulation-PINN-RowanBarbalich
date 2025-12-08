function plotXY(X,Y,Nx,Ny,text)
%% plots DeepCFD data
    % extract data
    objSDF = squeeze(X(:,:,1));
    masks = squeeze(X(:,:,2));
    wallSDF = squeeze(X(:,:,3));
    ux = squeeze(Y(:,:,1));
    uy = squeeze(Y(:,:,2));
    u = sqrt(ux.^2 + uy.^2); % compute resultant velocity
    P = squeeze(Y(:,:,3));
    object = masks==0; % store obstacle shape from masks
    objectRGB = ones(Ny,Nx,3); % draw obstacle in white
    numLines = floor(Ny/2); % number of streamlines

    % create mesh grid for streamlines
    x = 1:1:Nx;
    y = 1:1:Ny;
    [xm,ym] = meshgrid(x,y);

    % create plots
    figure()
    plot = tiledlayout(2,3,'TileSpacing','tight');
    title(plot,text)
    subtitle(plot,' ') % for extra spacing
    
        % object SDF
        nexttile(plot)
        hold on
        imagesc(objSDF')
        objImgSDF = image(0.5*objectRGB); % draw obstacle in grey (RGB = [0.5 0.5 0.5])
        set(objImgSDF,'AlphaData',object') % make obstacle background transparent
        title('SDF (object)')
        xlabel('Nx')
        ylabel('Ny')
        axis equal
        xlim([0 Nx])
        ylim([0 Ny])
        set(gca,'YDir','normal') % flip y-axis (first row is at the top by default)
        set(gca,'Colormap',turbo)
        colorbar
        clim([0 1])

        % masks
        nexttile(plot)
        imagesc(masks')
        title('Masks')
        xlabel('Nx')
        ylabel('Ny')
        axis equal
        xlim([0 Nx+0.5]) % +0.5 to show outlet and top wall masks
        ylim([0 Ny+0.5])
        set(gca,'YDir','normal')
        c = parula(5); % colorbar with 5 levels
        set(gca,'Colormap',c)
        colorbar('Ticks',0.4:0.8:4,'TickLabels',["Object" "Fluid" "Wall" "Inlet" "Outlet"]) % tick labels
    
        % wall SDF
        nexttile(plot)
        imagesc(wallSDF')
        title('SDF (walls)')
        xlabel('Nx')
        ylabel('Ny')
        axis equal
        xlim([0 Nx])
        ylim([0 Ny])
        set(gca,'YDir','normal')
        set(gca,'Colormap',turbo)
        colorbar
    
        % resultant velocity magnitude
        nexttile(plot)
        hold on
        imagesc(u')
        % streamline(xm,ym,ux',uy',ones(1,6),linspace(1,Ny,6),'Color','white','LineWidth',1) % plot 6 streamlines starting at the inlet
        objImgu = image(objectRGB);
        set(objImgu,'AlphaData',object')
        title('Resultant Velocity Magnitude')
        xlabel('Nx')
        ylabel('Ny')
        axis equal
        xlim([0 Nx])
        ylim([0 Ny])
        set(gca,'YDir','normal')
        set(gca,'Colormap',jet)
        cb = colorbar;
        cb.Label.String = 'U_{res} / U_{in}';

        % streamlines
        nexttile(plot)
        hold on
        objImgS = image(0.5*objectRGB); % plot object in gray
        set(objImgS,'AlphaData',object')
        streamline(xm,ym,ux',uy',ones(1,numLines),linspace(1,Ny,numLines)) % plot numLines streamlines starting at the inlet
        title('Streamlines')
        xlabel('Nx')
        ylabel('Ny')
        axis equal
        xlim([0 Nx])
        ylim([0 Ny])
        set(gca,'YDir','normal')
        set(gca,'Colormap',gray)
    
        % % pressure
        % nexttile(plot)
        % hold on
        % imagesc(P')
        % objImgP = image(objectRGB);
        % set(objImgP,'AlphaData',object')
        % title('Pressure')
        % xlabel('Nx')
        % ylabel('Ny')
        % axis equal
        % xlim([0 Nx])
        % ylim([0 Ny])
        % set(gca,'YDir','normal')
        % set(gca,'Colormap',jet)
        % colorbar

    set(gcf,'Visible','on') % open plots in new window
end