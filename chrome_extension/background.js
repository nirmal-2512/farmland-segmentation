chrome.runtime.onInstalled.addListener(() => {
  console.log('Farmland Boundary Detector installed.');
});

chrome.action.onClicked.addListener((tab) => {
  if (!tab.url.includes('google.com/maps')) {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icon48.png',
      title: 'Farmland Boundary Detector',
      message: 'Please open Google Maps first.'
    });
  }
});
