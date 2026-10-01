import './style.css';

// DOM Elements
const img1 = document.getElementById('img-1');
const prompt1 = document.getElementById('prompt-1');
const img2 = document.getElementById('img-2');
const prompt2 = document.getElementById('prompt-2');

const upload1 = document.getElementById('upload-1');
const upload2 = document.getElementById('upload-2');

const btnCompare = document.getElementById('btn-compare');
const btnReset = document.getElementById('reset-workspace');

const compResults = document.getElementById('comparison-results');
const resLoading = document.getElementById('results-loading');
const resComplete = document.getElementById('results-complete');

const valDesign = document.getElementById('val-design');
const valColor = document.getElementById('val-color');
const verdictTitle = document.getElementById('verdict-title');
const verdictDesc = document.getElementById('verdict-desc');

// Spec Elements
const specMatA = document.getElementById('spec-mat-a');
const specMatB = document.getElementById('spec-mat-b');
const specColA = document.getElementById('spec-col-a');
const specColB = document.getElementById('spec-col-b');
const specPatA = document.getElementById('spec-pat-a');
const specPatB = document.getElementById('spec-pat-b');

// State
let selectedImage1 = null;
let selectedImage2 = null;
let activeSlot = 1; // 1 or 2

// Reset Workspace
btnReset.addEventListener('click', () => {
  img1.src = '';
  img1.classList.add('hidden');
  prompt1.classList.remove('hidden');
  
  img2.src = '';
  img2.classList.add('hidden');
  prompt2.classList.remove('hidden');
  
  selectedImage1 = null;
  selectedImage2 = null;
  activeSlot = 1;
  
  btnCompare.disabled = true;
  compResults.classList.add('hidden');
  resLoading.classList.add('hidden');
  resComplete.classList.add('hidden');
  
  document.querySelectorAll('.gallery-item img').forEach(el => el.classList.remove('selected'));
});

// Upload Logic
function handleFileSelection(file, imgEl, promptEl, slotNum) {
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const src = e.target.result;
      imgEl.src = src;
      imgEl.classList.remove('hidden');
      promptEl.classList.add('hidden');
      
      if (slotNum === 1) {
        selectedImage1 = src;
        activeSlot = 2; // Auto-switch to next slot
      } else {
        selectedImage2 = src;
        activeSlot = 1;
      }
      
      if (selectedImage1 && selectedImage2) {
        btnCompare.disabled = false;
      }
    };
    reader.readAsDataURL(file);
  }
}

// Input Change
upload1?.addEventListener('change', (e) => {
  handleFileSelection(e.target.files[0], img1, prompt1, 1);
});
upload2?.addEventListener('change', (e) => {
  handleFileSelection(e.target.files[0], img2, prompt2, 2);
});

// Drag and Drop
[prompt1, prompt2].forEach((prompt, idx) => {
  if (!prompt) return;
  const slotNum = idx + 1;
  const imgEl = slotNum === 1 ? img1 : img2;
  
  prompt.addEventListener('dragover', (e) => {
    e.preventDefault();
    prompt.classList.add('drag-active');
  });
  
  prompt.addEventListener('dragleave', (e) => {
    e.preventDefault();
    prompt.classList.remove('drag-active');
  });
  
  prompt.addEventListener('drop', (e) => {
    e.preventDefault();
    prompt.classList.remove('drag-active');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0], imgEl, prompt, slotNum);
    }
  });
});


// Analyze Logic
const materials = ["Kanjeevaram Silk", "Banarasi Silk", "Cotton Blend", "Georgette", "Chiffon"];
const colors = ["Ruby Red", "Emerald Green", "Sapphire Blue", "Midnight Black", "Golden Yellow", "Crimson", "Navy", "Magenta"];
const patterns = ["Floral Paisley", "Zari Grid", "Diamond Lattice", "Peacock Motifs", "Temple Borders"];

btnCompare.addEventListener('click', () => {
  if (!selectedImage1 || !selectedImage2) return;
  
  btnCompare.disabled = true;
  compResults.classList.remove('hidden');
  resLoading.classList.remove('hidden');
  resComplete.classList.add('hidden');
  
  // Fake Inference Pipeline Simulation
  setTimeout(() => {
    resLoading.classList.add('hidden');
    resComplete.classList.remove('hidden');
    btnCompare.disabled = false;
    
    // Randomize output slightly for effect
    const designSim = parseFloat((Math.random() * (0.999 - 0.820) + 0.820).toFixed(3));
    const colorSim = (Math.random() * (0.3 - 0.05) + 0.05).toFixed(3);
    
    valDesign.textContent = designSim;
    valColor.textContent = colorSim;
    
    // Populate Specs
    specMatA.textContent = materials[Math.floor(Math.random() * materials.length)];
    specMatB.textContent = materials[Math.floor(Math.random() * materials.length)];
    specColA.textContent = colors[Math.floor(Math.random() * colors.length)];
    specColB.textContent = colors[Math.floor(Math.random() * colors.length)];
    
    // Pattern matches if design sim > 0.9
    const pat = patterns[Math.floor(Math.random() * patterns.length)];
    specPatA.textContent = pat;
    if (designSim > 0.9) {
      specPatB.textContent = pat;
      verdictTitle.textContent = "Match Confirmed";
      verdictTitle.style.color = "var(--success)";
      verdictDesc.textContent = "These sarees share the identical geometric structure despite color variations.";
    } else {
      specPatB.textContent = patterns[Math.floor(Math.random() * patterns.length)];
      verdictTitle.textContent = "Different Designs";
      verdictTitle.style.color = "#ef4444";
      verdictDesc.textContent = "The underlying structural manifolds do not align.";
    }
    
  }, 2000); 
});

// Gallery Population Logic
const galleryGrid = document.getElementById('gallery-grid');
const imageIds = [
  0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
  40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 59, 60, 61, 
  62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 77, 79
];

if (galleryGrid) {
  imageIds.forEach(id => {
    const imgWrapper = document.createElement('div');
    imgWrapper.className = 'gallery-item';
    
    const img = document.createElement('img');
    img.src = `/indian_saree/${id}.png`;
    img.alt = `Saree ${id}`;
    img.loading = 'lazy';
    
    imgWrapper.appendChild(img);
    galleryGrid.appendChild(imgWrapper);
    
    // Click to load into comparison slots
    imgWrapper.addEventListener('click', () => {
      if (activeSlot === 1) {
        // clear old selection if both were filled
        if (selectedImage1 && selectedImage2) {
          document.querySelectorAll('.gallery-item img').forEach(el => el.classList.remove('selected'));
        }
      }
      
      img.classList.add('selected');

      if (activeSlot === 1) {
        selectedImage1 = img.src;
        img1.src = selectedImage1;
        img1.classList.remove('hidden');
        prompt1.classList.add('hidden');
        activeSlot = 2; // Auto-switch to next slot
      } else {
        selectedImage2 = img.src;
        img2.src = selectedImage2;
        img2.classList.remove('hidden');
        prompt2.classList.add('hidden');
        activeSlot = 1; // loop back
      }
      
      // Enable comparison if both selected
      if (selectedImage1 && selectedImage2) {
        btnCompare.disabled = false;
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });
}

// AI Chatbot Logic
const btnOpenChat = document.getElementById('btn-open-chat');
const btnCloseChat = document.getElementById('btn-close-chat');
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const btnSendChat = document.getElementById('btn-send-chat');
const chatBody = document.getElementById('chat-body');

if (btnOpenChat && chatWindow && btnCloseChat) {
  btnOpenChat.addEventListener('click', () => {
    chatWindow.classList.remove('hidden');
    btnOpenChat.style.display = 'none';
  });

  btnCloseChat.addEventListener('click', () => {
    chatWindow.classList.add('hidden');
    btnOpenChat.style.display = 'flex';
  });

  function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // User Message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.textContent = text;
    chatBody.appendChild(userMsg);
    
    chatInput.value = '';
    chatBody.scrollTop = chatBody.scrollHeight;

    // Fake Bot Response after a small delay
    setTimeout(() => {
      const botMsg = document.createElement('div');
      botMsg.className = 'chat-message bot';
      botMsg.textContent = "I'm currently operating in offline demo mode. My full generative capabilities will be unlocked in the production release!";
      chatBody.appendChild(botMsg);
      chatBody.scrollTop = chatBody.scrollHeight;
    }, 1000);
  }

  btnSendChat.addEventListener('click', sendMessage);
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
}

// Sidebar Nav Logic
const navGallery = document.getElementById('nav-gallery');
const navAnalytics = document.getElementById('nav-analytics');
const analyticsModal = document.getElementById('analytics-modal');
const btnCloseAnalytics = document.getElementById('btn-close-analytics');

if (navGallery) {
  navGallery.addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('gallery-section').scrollIntoView({ behavior: 'smooth' });
  });
}

if (navAnalytics && analyticsModal && btnCloseAnalytics) {
  navAnalytics.addEventListener('click', (e) => {
    e.preventDefault();
    analyticsModal.classList.remove('hidden');
  });

  btnCloseAnalytics.addEventListener('click', () => {
    analyticsModal.classList.add('hidden');
  });

  analyticsModal.addEventListener('click', (e) => {
    if (e.target === analyticsModal) {
      analyticsModal.classList.add('hidden');
    }
  });
}
