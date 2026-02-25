// url base where se ejecuta the API of the model of classification
// localhost port 8000 es where fastapi serves the model resnet50
const API_BASE_URL = 'http://localhost:8000';

// objeto that mantiene the status global of the aplicacion
// centraliza toda the informacion importante durante the uso
const state = {
    currentImage: null,     // file of image actualmente seleccionado
    isLoading: false,      // indica if hay a operacion en progress
    results: null          // almacena the resultados of the ultima classification
};

// referencias a elementos of the dom for manipulacion eficiente
// avoids search elementos repetidamente with getelementbyid
const elements = {
    uploadArea: null,        // zone where se arrastra or selecciona images
    fileInput: null,         // input oculto that maneja selection of files
    imagePreview: null,      // contenedor that shows preview of image
    previewImg: null,        // elemento img that renderiza the image subida
    resetBtn: null,          // boton for limpiar and upload new image
    loadingContainer: null,  // indicador visual of processing
    resultsContainer: null   // area where se show resultados of classification
};

// evento that se ejecuta automaticamente when the dom this completamente cargado
// garantiza that all the elementos html esten disponibles antes of manipularlos
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();    // gets referencias a all the elementos necesarios
    setupEventListeners();   // configura all the manejadores of eventos
    initParticles();        // inicia the animacion of particulas en the background
    checkAPIConnection();    // verifies that the API of the model this disponible
});

// gets referencias a all the elementos html importantes and the guarda
// esto optimiza the performance evitando busquedas repetidas en the dom
function initializeElements() {
    elements.uploadArea = document.getElementById('uploadArea');           // zone of drag and drop
    elements.fileInput = document.getElementById('fileInput');             // input for seleccionar files
    elements.imagePreview = document.getElementById('imagePreview');       // contenedor of vista previa
    elements.previewImg = document.getElementById('previewImg');           // image mostrada to the user
    elements.resetBtn = document.getElementById('resetBtn');               // boton of reinicio
    elements.loadingContainer = document.getElementById('loadingContainer'); // indicador of load
    elements.resultsContainer = document.getElementById('resultsContainer'); // area of resultados
}

// configura all the manejadores of eventos for interaccion of the user
// establece como the interfaz responde a clics, drag and drop, etc
function setupEventListeners() {
    // evento of clic en zone of upload activa the selector of files
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    
    // eventos of drag and drop for drag images from the explorador
    elements.uploadArea.addEventListener('dragover', handleDragOver);   // when se arrastra about area
    elements.uploadArea.addEventListener('dragleave', handleDragLeave); // when se sale of the area
    elements.uploadArea.addEventListener('drop', handleDrop);           // when se suelta file

    // evento when user selecciona file manualmente
    elements.fileInput.addEventListener('change', handleFileSelect);

    // evento of the boton for resetear and upload new image
    elements.resetBtn.addEventListener('click', resetInterface);

    // previene comportamiento by default of the navegador en drag and drop
    // necesario for that funcione correctamente the funcionalidad personalizada
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, preventDefaults);
        document.body.addEventListener(eventName, preventDefaults);
    });
}

// previene the acciones by default of the navegador for eventos of arrastre
// necesario for implementar drag and drop personalizado
function preventDefaults(e) {
    e.preventDefault();  // cancela the accion by default of the navegador
    e.stopPropagation(); // avoids that the evento se propague a elementos padre
}

// maneja when a file se arrastra about the zone of upload
// changes the estilo visual for indicar zone activa
function handleDragOver(e) {
    elements.uploadArea.classList.add('dragover'); // applies estilo visual of zone activa
}

// maneja when the file sale of the zone of upload
// remueve the estilo visual of zone activa
function handleDragLeave(e) {
    elements.uploadArea.classList.remove('dragover'); // removes estilo of zone activa
}

// maneja when se suelta a file en the zone of upload
// procesa the primer file arrastrado
function handleDrop(e) {
    elements.uploadArea.classList.remove('dragover'); // removes estilo visual
    const files = e.dataTransfer.files;               // gets files arrastrados
    if (files.length > 0) {
        processFile(files[0]); // procesa only the primer file
    }
}

// maneja when user selecciona file with the boton
// procesa the file seleccionado of the input
function handleFileSelect(e) {
    const file = e.target.files[0]; // gets primer file seleccionado
    if (file) {
        processFile(file); // procesa the file
    }
}

// procesa and valid the file seleccionado antes of enviarlo a classification
// realiza verificaciones of seguridad and format antes of continuar
function processFile(file) {
    // verifies that the file sea realmente a image
    // startswith verifies the tipo mime of the file
    if (!file.type.startsWith('image/')) {
        showError('Por favor selecciona un archivo de imagen válido');
        return; // termina ejecucion if no es image
    }

    // Implementation note.
    // 10 * 1024 * 1024 = 10485760 bytes = 10mb
    if (file.size > 10 * 1024 * 1024) {
        showError('La imagen es demasiado grande. Máximo 10MB');
        return; // termina ejecucion if es very grande
    }

    state.currentImage = file;    // guarda referencia to the file en status global
    displayImagePreview(file);    // shows preview of the image to the user
    classifyImage(file);         // inicia process of classification with API
}

// shows a vista previa of the image seleccionada antes of procesarla
// convierte the file en a url that can show the navegador
function displayImagePreview(file) {
    const reader = new FileReader(); // API of the navegador for leer files
    
    // evento that se ejecuta when filereader termina of leer the file
    reader.onload = function(e) {
        elements.previewImg.src = e.target.result; // asigna data of image to the elemento img
        showSection('imagePreview');               // hace visible the seccion of preview
        hideSection('uploadArea');                 // hidden the zone of upload original
    };
    
    // inicia the lectura of the file como data url base64
    // esto convierte the image en texto that the navegador can show
    reader.readAsDataURL(file);
}

// envia the image a the API of the model for classification of breed
// maneja todo the process of comunicacion with the backend
async function classifyImage(file) {
    try {
        showSection('loadingContainer');  // shows indicador of load to the user
        hideSection('resultsContainer');  // hidden resultados anteriores if existen
        state.isLoading = true;          // marca status global como procesando

        // prepara data for envio multipart form data requerido for fastapi
        const formData = new FormData();
        formData.append('file', file);   // adjunta file with name 'file'

        console.log('🔄 Enviando imagen a la API...'); // log for debugging
        
        // realiza peticion http post to the endpoint of prediction
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData  // no se especifica content-type, fetch lo hace automaticamente
        });

        // verifies if the respuesta http fue exitosa
        if (!response.ok) {
            const errorText = await response.text(); // Implementation note.
            console.error('Error de la API:', errorText);
            throw new Error(`Error del servidor: ${response.status} - ${errorText}`);
        }

        // convierte respuesta json en objeto javascript
        const results = await response.json();
        console.log('✅ Respuesta de la API:', results); // log for debugging
        
        // verifies that the API devolvio a respuesta valid
        if (!results || !results.success) {
            throw new Error('La API devolvió una respuesta de error');
        }

        state.results = results;  // guarda resultados en status global
        displayResults(results);  // shows resultados to the user

    } catch (error) {
        console.error('❌ Error en clasificación:', error);
        showError(`Error al clasificar la imagen: ${error.message}`);
    } finally {
        state.isLoading = false;         // marca that process termino
        hideSection('loadingContainer'); // hidden indicador of load
    }
}

// construye and shows the interfaz of resultados with the predictions of the model
// creates dinamicamente elementos html for a presentacion visual atractiva
function displayResults(results) {
    const container = elements.resultsContainer;
    
    container.innerHTML = ''; // limpia cualquier resultado anterior

    // verifies that the data of the API tengan the format correcto
    // avoids errors if the respuesta this mal formada
    if (!results || !results.top_predictions || !Array.isArray(results.top_predictions)) {
        showError('Error: Formato de respuesta de la API inválido');
        return; // termina ejecucion if data son invalidos
    }

    // creates the contenedor main for the resultados
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card'; // applies estilos css

    // creates seccion of encabezado with resultado main
    const header = document.createElement('div');
    header.className = 'result-header';
    
    // elemento for show the breed more probable
    const title = document.createElement('h2');
    title.className = 'result-title';
    
    // gets the breed main of the respuesta with fallback seguro
    const mainBreed = results.recommendation?.most_likely || results.top_predictions[0]?.breed || 'Desconocido';
    title.textContent = `🐕 ${formatBreedName(mainBreed)}`;
    
    // creates badge of confianza with color segun nivel
    const confidenceBadge = document.createElement('div');
    
    // gets confianza main with fallback seguro
    const mainConfidence = results.recommendation?.confidence || results.top_predictions[0]?.confidence || 0;
    confidenceBadge.className = `confidence-badge ${getConfidenceLevel(mainConfidence)}`;
    confidenceBadge.textContent = `${(mainConfidence * 100).toFixed(1)}% confianza`;
    
    // ensambla the encabezado
    header.appendChild(title);
    header.appendChild(confidenceBadge);

    // creates list of the best predictions
    const predictionsList = document.createElement('div');
    predictionsList.className = 'predictions-list';

    // shows only the 5 best predictions for no saturar interfaz
    const topPredictions = results.top_predictions.slice(0, 5);
    topPredictions.forEach((prediction, index) => {
        const item = createPredictionItem(prediction, index + 1); // index + 1 for ranking humano
        predictionsList.appendChild(item);
    });

    // ensambla toda the tarjeta of resultados
    resultCard.appendChild(header);
    resultCard.appendChild(predictionsList);
    container.appendChild(resultCard);

    showSection('resultsContainer'); // hace visible the seccion of resultados
}

// creates a elemento visual individual for cada prediction of breed
// incluye ranking, name, bar of confianza and porcentaje
function createPredictionItem(prediction, rank) {
    // contenedor main of the item with estilo especial for the primero
    const item = document.createElement('div');
    item.className = `prediction-item ${rank === 1 ? 'top' : ''}`; // class especial for #1

    // seccion izquierda with ranking and name of breed
    const breedInfo = document.createElement('div');
    breedInfo.className = 'breed-info';

    // badge circular with number of ranking
    const rankBadge = document.createElement('div');
    rankBadge.className = 'breed-rank';
    rankBadge.textContent = rank; // shows 1, 2, 3, etc

    // name of the breed formateado for lectura humana
    const breedName = document.createElement('div');
    breedName.className = 'breed-name';
    breedName.textContent = formatBreedName(prediction.breed);

    // ensambla seccion of informacion of breed
    breedInfo.appendChild(rankBadge);
    breedInfo.appendChild(breedName);

    // seccion derecha with visualizacion of confianza
    const confidenceInfo = document.createElement('div');
    confidenceInfo.className = 'confidence-info';

    // contenedor of bar of progress
    const confidenceBar = document.createElement('div');
    confidenceBar.className = 'confidence-bar';

    // bar of progress that se llena segun porcentaje of confianza
    const confidenceFill = document.createElement('div');
    confidenceFill.className = 'confidence-fill';
    confidenceFill.style.width = `${prediction.confidence * 100}%`; // ancho proporcional

    confidenceBar.appendChild(confidenceFill);

    // texto with porcentaje numerico
    const confidencePercent = document.createElement('div');
    confidencePercent.className = 'confidence-percent';
    confidencePercent.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    // ensambla seccion of confianza
    confidenceInfo.appendChild(confidenceBar);
    confidenceInfo.appendChild(confidencePercent);

    // ensambla item complete
    item.appendChild(breedInfo);
    item.appendChild(confidenceInfo);

    return item; // returns elemento listo for insertar en dom
}

// convierte names tecnicos of breeds en format readable for humans
// transforma 'golden_retriever' en 'Golden Retriever'
function formatBreedName(breedName) {
    return breedName
        .split('_')                                    // separa for guiones bajos
        .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // capitaliza cada palabra
        .join(' ');                                    // une with espacios
}

// determina the nivel of confianza for apply colores apropiados
// ayuda to the user a interpretar rapidamente that tan seguro this the model
function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'high';    // verde for alta confianza
    if (confidence >= 0.5) return 'medium';  // amarillo for confianza media
    return 'low';                             // rojo for low confianza
}

// resets toda the interfaz to the status initial for new classification
// limpia data anteriores and restaura vista of upload
function resetInterface() {
    state.currentImage = null;  // borra referencia to the file anterior
    state.results = null;       // borra resultados anteriores
    
    elements.fileInput.value = '';    // limpia input of file
    elements.previewImg.src = '';     // borra image mostrada
    
    // controla visibilidad of secciones
    hideSection('imagePreview');      // hidden preview of image
    hideSection('loadingContainer');  // hidden indicador of load
    hideSection('resultsContainer');  // hidden resultados anteriores
    showSection('uploadArea');        // shows zone of upload original
}

// hace visible a seccion especifica of the interfaz
// centraliza the control of visibilidad for consistency
function showSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'block'; // hace visible with display block
    }
}

// hidden a seccion especifica of the interfaz
// centraliza the control of visibilidad for consistency
function hideSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'none'; // hidden with display none
    }
}

// Implementation note.
// creates elementos temporales that aparecen and desaparecen automaticamente
function showError(message) {
    // creates elemento of notificacion toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    
    // applies estilos css inline for posicionamiento and apariencia
    toast.style.cssText = `
        position: fixed;           /* posicion fija en viewport */
        top: 20px;                /* 20px from arriba */
        right: 20px;              /* 20px from derecha */
        background: #ef4444;      /* Implementation note. */
        color: white;             /* texto blanco */
        padding: 1rem 1.5rem;     /* espaciado interno */
        border-radius: 12px;      /* bordes redondeados */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); /* sombra */
        z-index: 1000;            /* for encima of otros elementos */
        animation: slideInRight 0.3s ease-out; /* animacion of input */
        max-width: 400px;         /* ancho maximo */
        font-weight: 500;         /* peso of fuente */
    `;
    toast.textContent = message; // contenido of the mensaje

    // creates estilos of animacion dinamicamente
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%); /* empieza fuera of pantalla */
                opacity: 0;
            }
            to {
                transform: translateX(0);    /* termina en posicion normal */
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style); // adds estilos to the documento

    document.body.appendChild(toast); // adds toast to the documento

    // remueve automaticamente the toast despues of 5 segundos
    setTimeout(() => {
        toast.remove();  // elimina toast of the dom
        style.remove();  // elimina estilos of the dom
    }, 5000);
}

// verifies that the API of the model this disponible and respondiendo
// realiza a health check to the load the pagina for detect problemas temprano
async function checkAPIConnection() {
    try {
        // peticion simple to the endpoint of salud of the API
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            console.log('✅ Conexión con API establecida'); // confirmacion en consola
        } else {
            throw new Error('API no disponible'); // Implementation note.
        }
    } catch (error) {
        // informa to the user if hay problemas of conectividad
        console.warn('⚠️ No se pudo conectar con la API:', error.message);
        showError('No se pudo conectar con el servidor. Asegúrate de que la API esté ejecutándose en el puerto 8000');
    }
}

// initializes the system of particulas animadas en the background
// creates a efecto visual atractivo without interferir with the funcionalidad
function initParticles() {
    const canvas = document.getElementById('particles');   // gets canvas for dibujo
    const ctx = canvas.getContext('2d');                  // contexto 2d for dibujar
    
    // Implementation note.
    function resizeCanvas() {
        canvas.width = window.innerWidth;   // ancho igual to the viewport
        canvas.height = window.innerHeight; // alto igual to the viewport
    }
    
    resizeCanvas(); // Implementation note.
    window.addEventListener('resize', resizeCanvas); // reajusta to the cambiar ventana

    // configuration of the system of particulas
    const particles = [];           // array that contiene all the particulas
    const particleCount = 50;       // number total of particulas

    // creates particulas iniciales with propiedades aleatorias
    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,          // posicion x aleatoria
            y: Math.random() * canvas.height,         // posicion and aleatoria
            size: Math.random() * 3 + 1,              // Implementation note.
            speedX: (Math.random() - 0.5) * 0.5,      // velocidad horizontal
            speedY: (Math.random() - 0.5) * 0.5,      // velocidad vertical
            opacity: Math.random() * 0.5 + 0.2        // opacidad entre 0.2 and 0.7
        });
    }

    // function of animacion that se ejecuta continuamente
    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height); // limpia frame anterior

        // updates and dibuja cada particula
        particles.forEach(particle => {
            // updates posicion basada en velocidad
            particle.x += particle.speedX;
            particle.y += particle.speedY;

            // rebote en bordes of pantalla
            if (particle.x < 0 || particle.x > canvas.width) particle.speedX *= -1;
            if (particle.y < 0 || particle.y > canvas.height) particle.speedY *= -1;

            // dibuja particula como circulo blanco semitransparente
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${particle.opacity})`;
            ctx.fill();
        });

        requestAnimationFrame(animateParticles); // programa siguiente frame
    }

    animateParticles(); // inicia loop of animacion
}

// convierte bytes a format readable for humans
// Implementation note.
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';  // caso especial for files vacios
    
    const k = 1024;                     // factor of conversion
    const sizes = ['Bytes', 'KB', 'MB', 'GB']; // unidades disponibles
    
    // Implementation note.
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    // formatea number with 2 decimales and adds unidad apropiada
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// gets informacion detailed of the model from the API
// util for debugging and verificacion of configuration
async function getModelInfo() {
    try {
        // peticion to the endpoint of informacion of the model
        const response = await fetch(`${API_BASE_URL}/model-info`);
        
        if (response.ok) {
            const info = await response.json();
            console.log('📊 Información del modelo:', info); // shows en consola
            return info; // returns data for uso posterior
        }
    } catch (error) {
        console.warn('No se pudo obtener información del modelo:', error);
    }
    return null; // Implementation note.
}