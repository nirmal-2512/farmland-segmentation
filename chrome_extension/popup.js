const captureButton = document.getElementById('captureButton');
const clearOverlayButton = document.getElementById('clearOverlayButton');
const downloadButton = document.getElementById('downloadButton');
const downloadKmlButton = document.getElementById('downloadKmlButton');
const statusText = document.getElementById('status');
const geojsonText = document.getElementById('geojsonText');

let latestGeoJSON = null;

function updateStatus(message) {
  statusText.textContent = `Status: ${message}`;
}

function convertGeoJSONToKml(geojson) {
  const kmlParts = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<kml xmlns="http://www.opengis.net/kml/2.2">',
    '<Document>',
    '<name>Farmland Boundaries</name>'
  ];

  geojson.features.forEach((feature, index) => {
    const coords = feature.geometry.coordinates[0]
      .map(coord => `${coord[0]},${coord[1]},0`)
      .join(' ');

    const name = feature.properties && feature.properties.index != null
      ? `Field ${feature.properties.index}`
      : `Field ${index}`;

    kmlParts.push(
      '<Placemark>',
      `<name>${name}</name>`,
      '<Polygon>',
      '<outerBoundaryIs>',
      '<LinearRing>',
      `<coordinates>${coords}</coordinates>`,
      '</LinearRing>',
      '</outerBoundaryIs>',
      '</Polygon>',
      '</Placemark>'
    );
  });

  kmlParts.push('</Document>', '</kml>');
  return kmlParts.join('');
}

async function executeTabScript(tabId, func, args = []) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId },
      func,
      args
    });
    if (!results || !results[0]) {
      return { success: false, message: 'No script result' };
    }
    return results[0].result;
  } catch (err) {
    return { success: false, message: err.message || String(err) };
  }
}

async function getMapStateViaScripting(tabId) {
  return executeTabScript(tabId, () => {
    function parseUrlCenterZoom(url) {
      const atMatch = url.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)z/);
      if (atMatch) {
        return {
          lat: parseFloat(atMatch[1]),
          lng: parseFloat(atMatch[2]),
          zoom: parseFloat(atMatch[3])
        };
      }

      const llMatch = url.match(/[?&](?:center|ll)=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/);
      const zoomMatch = url.match(/[?&](?:zoom|z)=(\d+(?:\.\d+)?)/);
      if (llMatch && zoomMatch) {
        return {
          lat: parseFloat(llMatch[1]),
          lng: parseFloat(llMatch[2]),
          zoom: parseFloat(zoomMatch[1])
        };
      }

      const altMatch = url.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)(m|km)(?:[/?]|$)/);
      if (altMatch) {
        const meters = parseFloat(altMatch[3]) * (altMatch[4] === 'km' ? 1000 : 1);
        const estimatedZoom = Math.max(1, Math.min(21, 19 - Math.round(Math.log10(meters) - 2)));
        return {
          lat: parseFloat(altMatch[1]),
          lng: parseFloat(altMatch[2]),
          zoom: estimatedZoom
        };
      }

      const earthMatch = url.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)(m|km)a(?:,|$)/);
      if (earthMatch) {
        const meters = parseFloat(earthMatch[3]) * (earthMatch[4] === 'km' ? 1000 : 1);
        const estimatedZoom = Math.max(1, Math.min(21, 19 - Math.round(Math.log10(meters) - 2)));
        return {
          lat: parseFloat(earthMatch[1]),
          lng: parseFloat(earthMatch[2]),
          zoom: estimatedZoom
        };
      }

      const gmapMatch = url.match(/!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)/);
      if (gmapMatch) {
        return {
          lat: parseFloat(gmapMatch[1]),
          lng: parseFloat(gmapMatch[2]),
          zoom: 18
        };
      }

      const hashMatch = url.match(/#@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?)z/);
      if (hashMatch) {
        return {
          lat: parseFloat(hashMatch[1]),
          lng: parseFloat(hashMatch[2]),
          zoom: parseFloat(hashMatch[3])
        };
      }

      return null;
    }

    function getMapRect() {
      const canvasCandidates = Array.from(document.querySelectorAll('canvas'))
          .map(canvas => ({
              canvas,
              rect: canvas.getBoundingClientRect()
          }))
          .filter(entry => entry.rect.width > 200 && entry.rect.height > 200);
  
      if (canvasCandidates.length) {
          const best = canvasCandidates.reduce((prev, curr) => {
              const prevArea = prev.rect.width * prev.rect.height;
              const currArea = curr.rect.width * curr.rect.height;
              return currArea > prevArea ? curr : prev;
          });
  
          const rect = best.rect;
  
          // ── NEW: detect and exclude left sidebar ──────────
          // Google Maps sidebar is always a nav element on the left
          const sidebar = document.querySelector(
              '[class*="app-vertical-widget"], [class*="sidenav"], nav'
          );
          const sidebarWidth = sidebar
              ? Math.ceil(sidebar.getBoundingClientRect().right)
              : 80; // fallback — sidebar is ~80px
  
          return {
              left: sidebarWidth,
              top: rect.top,
              width: rect.width - sidebarWidth,
              height: rect.height
          };
      }
  
      const sidebarWidth = 80;
      return {
          left: sidebarWidth,
          top: 0,
          width: window.innerWidth - sidebarWidth,
          height: window.innerHeight
      };
  }

    function lonToPixel(lon, zoom) {
      const x = (lon + 180) / 360;
      const worldSize = 256 * Math.pow(2, zoom);
      return x * worldSize;
    }

    function latToPixel(lat, zoom) {
      const sinLat = Math.sin((lat * Math.PI) / 180);
      const y = 0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI);
      const worldSize = 256 * Math.pow(2, zoom);
      return y * worldSize;
    }

    function pixelToLon(x, zoom) {
      const worldSize = 256 * Math.pow(2, zoom);
      return (x / worldSize) * 360 - 180;
    }

    function pixelToLat(y, zoom) {
      const worldSize = 256 * Math.pow(2, zoom);
      const yNorm = 0.5 - (y / worldSize);
      return 90 - (360 * Math.atan(Math.exp(-yNorm * 2 * Math.PI))) / Math.PI;
    }

    const url = window.location.href;
    const centerZoom = parseUrlCenterZoom(url);
    const mapRect = getMapRect();

    if (!centerZoom) {
      return { success: false, message: 'Unable to parse map center/zoom from URL.' };
    }

    const { lat, lng, zoom } = centerZoom;

    // Use CSS pixel dimensions for bounds calculation
    // because lonToPixel/latToPixel work in CSS pixel world space
    const centerX = lonToPixel(lng, zoom);
    const centerY = latToPixel(lat, zoom);
    
    // mapRect.width/height are in CSS pixels — correct for bounds
    const halfWidth = mapRect.width / 2;
    const halfHeight = mapRect.height / 2;
    const nwX = centerX - halfWidth;
    const nwY = centerY - halfHeight;
    const seX = centerX + halfWidth;
    const seY = centerY + halfHeight;
    
    // DPR for physical pixel crop dimensions
    const dpr = window.devicePixelRatio || 1;
    
    // Physical pixel dimensions of the map area
    // These must match what captureVisibleTab actually captures
    const physicalWidth = Math.round(mapRect.width * dpr);
    const physicalHeight = Math.round(mapRect.height * dpr);
    
    return {
      success: true,
      url,
      center: { lat, lng },
      zoom,
      bounds: {
        north: pixelToLat(nwY, zoom),
        west: pixelToLon(nwX, zoom),
        south: pixelToLat(seY, zoom),
        east: pixelToLon(seX, zoom)
      },
      // mapRect in CSS pixels — used for overlay drawing
      mapRect: {
        left: mapRect.left,
        top: mapRect.top,
        width: mapRect.width,
        height: mapRect.height
      },
      // Physical pixel dimensions — used for crop and sent to FastAPI
      physicalWidth,
      physicalHeight,
      devicePixelRatio: dpr,
      pageTitle: document.title
    };
  });
}

async function drawOverlayViaScripting(tabId, geojson, mapRect, bounds) {
  return executeTabScript(tabId, (geojsonData, mapRectData, geoBounds) => {
    const existing = document.getElementById('farmboundary-overlay-container');
    if (existing) {
      existing.remove();
    }

    const rect = mapRectData && mapRectData.width > 0 && mapRectData.height > 0
      ? mapRectData
      : { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };

    const container = document.createElement('div');
    container.id = 'farmboundary-overlay-container';
    container.style.position = 'fixed';
    container.style.top = `${rect.top}px`;
    container.style.left = `${rect.left}px`;
    container.style.width = `${rect.width}px`;
    container.style.height = `${rect.height}px`;
    container.style.pointerEvents = 'none';
    container.style.zIndex = '999999999';

    const canvas = document.createElement('canvas');
    canvas.id = 'farmboundary-overlay-canvas';
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    canvas.style.pointerEvents = 'none';
    container.appendChild(canvas);
    document.body.appendChild(container);

    const rectWidth = rect.width;
    const rectHeight = rect.height;
    canvas.width = rectWidth * window.devicePixelRatio;
    canvas.height = rectHeight * window.devicePixelRatio;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, rectWidth, rectHeight);

    function latToMercatorY(lat) {
      const latRad = (lat * Math.PI) / 180;
      return Math.log(Math.tan(Math.PI / 4 + latRad / 2));
    }
    
    const northY = latToMercatorY(geoBounds.north);
    const southY = latToMercatorY(geoBounds.south);
    
    function projectPoint(lng, lat) {
      // Longitude is linear
      const x = ((lng - geoBounds.west) / (geoBounds.east - geoBounds.west)) * rectWidth;
    
      // Latitude uses Mercator to match Google Maps tiles exactly
      const mercY = latToMercatorY(lat);
      const y = ((northY - mercY) / (northY - southY)) * rectHeight;
    
      return { x, y };
    }

    geojsonData.features.forEach(feature => {
      const coords = feature.geometry.coordinates[0].map(([lng, lat]) => projectPoint(lng, lat));
      if (coords.length < 2) return;

      ctx.beginPath();
      coords.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(255, 0, 0, 0.18)';
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
      ctx.lineWidth = 3;
      ctx.fill();
      ctx.stroke();
    });

    return { success: true };
  }, [geojson, mapRect, bounds]);
}

async function clearOverlayViaScripting(tabId) {
  return executeTabScript(tabId, () => {
    const existing = document.getElementById('farmboundary-overlay-container');
    if (existing) {
      existing.remove();
    }
    return { success: true };
  });
}

async function captureTileAndDetect() {
  updateStatus('Requesting map capture...');

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    updateStatus('No active tab found.');
    return;
  }

  if (!tab.url.includes('google.com/maps') && !tab.url.includes('earth.google.com') && !tab.url.includes('google.com/earth')) {
    updateStatus('Open Google Maps or Google Earth in current tab.');
    return;
  }

  updateStatus('Capturing screenshot...');

  try {
    const capture = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' });
    const blob = await (await fetch(capture)).blob();
    const mapState = await getMapStateViaScripting(tab.id);

    if (!mapState.success) {
      throw new Error(`Map state error: ${mapState.message} URL=${tab.url}`);
    }

    const imageBitmap = await createImageBitmap(blob);
    const dpr = mapState.devicePixelRatio || 1;
    let cropBlob = blob;
    let imageWidth = imageBitmap.width;
    let imageHeight = imageBitmap.height;

    if (mapState.mapRect && mapState.mapRect.width > 0 && mapState.mapRect.height > 0) {
      const dpr = mapState.devicePixelRatio || 1;
      const sx = Math.round(mapState.mapRect.left * dpr);
      const sy = Math.round(mapState.mapRect.top * dpr);
    
      // Use pre-calculated physical dimensions
      const sw = mapState.physicalWidth;
      const sh = mapState.physicalHeight;
    
      const canvas = document.createElement('canvas');
      canvas.width = sw;
      canvas.height = sh;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imageBitmap, sx, sy, sw, sh, 0, 0, sw, sh);
      cropBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    
      // Send physical pixel dimensions to FastAPI
      // so pixel_to_geo divides by the correct size
      imageWidth = sw;
      imageHeight = sh;
    }

    updateStatus('Sending image and bounds to FastAPI...');

    const formData = new FormData();
    formData.append('file', cropBlob, 'tile.png');
    formData.append('north', mapState.bounds.north);
    formData.append('south', mapState.bounds.south);
    formData.append('east', mapState.bounds.east);
    formData.append('west', mapState.bounds.west);
    formData.append('image_width', imageWidth);
    formData.append('image_height', imageHeight);
    formData.append('threshold', 0.25);
    formData.append('return_mask', false);

    const response = await fetch('http://localhost:8000/predict-georef', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Server error: ${response.status} ${text}`);
    }

    const result = await response.json();
    latestGeoJSON = result.geojson;
    geojsonText.value = JSON.stringify(latestGeoJSON, null, 2);
    downloadButton.disabled = false;
    downloadKmlButton.disabled = false;

    updateStatus('Detection complete. Drawing overlay...');
    const overlayResult = await drawOverlayViaScripting(tab.id, latestGeoJSON, mapState.mapRect, mapState.bounds);
    if (!overlayResult.success) {
      throw new Error(`Overlay error: ${overlayResult.message}`);
    }
    updateStatus('Detection complete. GeoJSON ready and overlay drawn.');
  } catch (error) {
    console.error(error);
    updateStatus(`Error: ${error.message}`);
  }
}

function downloadGeoJSON() {
  if (!latestGeoJSON) return;

  const blob = new Blob([JSON.stringify(latestGeoJSON, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'farmland_boundaries.geojson';
  anchor.click();
  URL.revokeObjectURL(url);
}

function downloadKML() {
  if (!latestGeoJSON) return;

  const kml = convertGeoJSONToKml(latestGeoJSON);
  const blob = new Blob([kml], { type: 'application/vnd.google-earth.kml+xml' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'farmland_boundaries.kml';
  anchor.click();
  URL.revokeObjectURL(url);
}

async function clearOverlay() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  const result = await clearOverlayViaScripting(tab.id);
  if (!result.success) {
    updateStatus(`Error: ${result.message}`);
    return;
  }
  updateStatus('Overlay cleared.');
}

captureButton.addEventListener('click', captureTileAndDetect);
clearOverlayButton.addEventListener('click', clearOverlay);
downloadButton.addEventListener('click', downloadGeoJSON);
downloadKmlButton.addEventListener('click', downloadKML);
