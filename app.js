// url base where se ejecuta la API of the model de classification
// localhost port 8000 es where fastapi sirve el model resnet50
const API_BASE_URL = 'http://localhost:8000';

// objeto that mantiene el estado global de la aplicacion
// centraliza toda la informacion importante durante el uso
const state = {
    currentImage: null,     // file de image actualmente seleccionado
    isLoading: false,      // indica if hay una operacion en progreso
    results: null          // almacena los resultados de la ultima classification
};

// referencias a elementos of the dom for manipulacion eficiente
// evita buscar elementos repetidamente with getelementbyid
const elements = {
    uploadArea: null,        // zona where se arrastra o selecciona images
    fileInput: null,         // input oculto that maneja seleccion de files
    imagePreview: null,      // contenedor that muestra preview de image
    previewImg: null,        // elemento img that renderiza la image subida
    resetBtn: null,          // boton for limpiar y subir nueva image
    loadingContainer: null,  // indicador visual de processing
    resultsContainer: null   // area where se muestran resultados de classification
};

// evento that se ejecuta automaticamente when el dom this completamente cargado
// garantiza that all los elementos html esten disponibles antes de manipularlos
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();    // obtiene referencias a all los elementos necesarios
    setupEventListeners();   // configura all los manejadores de eventos
    initParticles();        // inicia la animacion de particulas en el fondo
    checkAPIConnection();    // verifica that la API of the model this disponible
});

// obtiene referencias a all los elementos html importantes y los guarda
// esto optimiza el rendimiento evitando busquedas repetidas en el dom
function initializeElements() {
    elements.uploadArea = document.getElementById('uploadArea');           // zona de drag and drop
    elements.fileInput = document.getElementById('fileInput');             // input for seleccionar files
    elements.imagePreview = document.getElementById('imagePreview');       // contenedor de vista previa
    elements.previewImg = document.getElementById('previewImg');           // image mostrada to the user
    elements.resetBtn = document.getElementById('resetBtn');               // boton de reinicio
    elements.loadingContainer = document.getElementById('loadingContainer'); // indicador de load
    elements.resultsContainer = document.getElementById('resultsContainer'); // area de resultados
}

// configura all los manejadores de eventos for interaccion of the user
// establece como la interfaz responde a clics, drag and drop, etc
function setupEventListeners() {
    // evento de clic en zona de upload activa el selector de files
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    
    // eventos de drag and drop for arrastrar images desde el explorador
    elements.uploadArea.addEventListener('dragover', handleDragOver);   // when se arrastra sobre area
    elements.uploadArea.addEventListener('dragleave', handleDragLeave); // when se sale of the area
    elements.uploadArea.addEventListener('drop', handleDrop);           // when se suelta file

    // evento when user selecciona file manualmente
    elements.fileInput.addEventListener('change', handleFileSelect);

    // evento of the boton for resetear y subir nueva image
    elements.resetBtn.addEventListener('click', resetInterface);

    // previene comportamiento by default of the navegador en drag and drop
    // necesario for that funcione correctamente la funcionalidad personalizada
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, preventDefaults);
        document.body.addEventListener(eventName, preventDefaults);
    });
}

// previene las acciones by default of the navegador for eventos de arrastre
// necesario for implementar drag and drop personalizado
function preventDefaults(e) {
    e.preventDefault();  // cancela la accion by default of the navegador
    e.stopPropagation(); // evita that el evento se propague a elementos padre
}

// maneja when un file se arrastra sobre la zona de upload
// cambia el estilo visual for indicar zona activa
function handleDragOver(e) {
    elements.uploadArea.classList.add('dragover'); // aplica estilo visual de zona activa
}

// maneja when el file sale de la zona de upload
// remueve el estilo visual de zona activa
function handleDragLeave(e) {
    elements.uploadArea.classList.remove('dragover'); // quita estilo de zona activa
}

// maneja when se suelta un file en la zona de upload
// procesa el primer file arrastrado
function handleDrop(e) {
    elements.uploadArea.classList.remove('dragover'); // quita estilo visual
    const files = e.dataTransfer.files;               // obtiene files arrastrados
    if (files.length > 0) {
        processFile(files[0]); // procesa only el primer file
    }
}

// maneja when user selecciona file with el boton
// procesa el file seleccionado of the input
function handleFileSelect(e) {
    const file = e.target.files[0]; // obtiene primer file seleccionado
    if (file) {
        processFile(file); // procesa el file
    }
}

// procesa y valida el file seleccionado antes de enviarlo a classification
// realiza verificaciones de seguridad y formato antes de continuar
function processFile(file) {
    // verifica that el file sea realmente una image
    // startswith verifica el tipo mime of the file
    if (!file.type.startsWith('image/')) {
        showError('Por favor selecciona un archivo de imagen válido');
        return; // termina ejecucion if no es image
    }

    // Implementation note.
    // 10 * 1024 * 1024 = 10485760 bytes = 10mb
    if (file.size > 10 * 1024 * 1024) {
        showError('La imagen es demasiado grande. Máximo 10MB');
        return; // termina ejecucion if es muy grande
    }

    state.currentImage = file;    // guarda referencia to the file en estado global
    displayImagePreview(file);    // muestra preview de la image to the user
    classifyImage(file);         // inicia process de classification with API
}

// muestra una vista previa de la image seleccionada antes de procesarla
// convierte el file en una url that puede mostrar el navegador
function displayImagePreview(file) {
    const reader = new FileReader(); // API of the navegador for leer files
    
    // evento that se ejecuta when filereader termina de leer el file
    reader.onload = function(e) {
        elements.previewImg.src = e.target.result; // asigna data de image to the elemento img
        showSection('imagePreview');               // hace visible la seccion de preview
        hideSection('uploadArea');                 // oculta la zona de upload original
    };
    
    // inicia la lectura of the file como data url base64
    // esto convierte la image en texto that el navegador puede mostrar
    reader.readAsDataURL(file);
}

// envia la image a la API of the model for classification de breed
// maneja todo el process de comunicacion with el backend
async function classifyImage(file) {
    try {
        showSection('loadingContainer');  // muestra indicador de load to the user
        hideSection('resultsContainer');  // oculta resultados anteriores if existen
        state.isLoading = true;          // marca estado global como procesando

        // prepara data for envio multipart form data requerido por fastapi
        const formData = new FormData();
        formData.append('file', file);   // adjunta file with name 'file'

        console.log('🔄 Enviando imagen a la API...'); // log for debugging
        
        // realiza peticion http post to the endpoint de prediction
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData  // no se especifica content-type, fetch lo hace automaticamente
        });

        // verifica if la respuesta http fue exitosa
        if (!response.ok) {
            const errorText = await response.text(); // Implementation note.
            console.error('Error de la API:', errorText);
            throw new Error(`Error del servidor: ${response.status} - ${errorText}`);
        }

        // convierte respuesta json en objeto javascript
        const results = await response.json();
        console.log('✅ Respuesta de la API:', results); // log for debugging
        
        // verifica that la API devolvio una respuesta valida
        if (!results || !results.success) {
            throw new Error('La API devolvió una respuesta de error');
        }

        state.results = results;  // guarda resultados en estado global
        displayResults(results);  // muestra resultados to the user

    } catch (error) {
        console.error('❌ Error en clasificación:', error);
        showError(`Error al clasificar la imagen: ${error.message}`);
    } finally {
        state.isLoading = false;         // marca that process termino
        hideSection('loadingContainer'); // oculta indicador de load
    }
}

// construye y muestra la interfaz de resultados with las predictions of the model
// crea dinamicamente elementos html for una presentacion visual atractiva
function displayResults(results) {
    const container = elements.resultsContainer;
    
    container.innerHTML = ''; // limpia cualquier resultado anterior

    // verifica that los data de la API tengan el formato correcto
    // evita errors if la respuesta this mal formada
    if (!results || !results.top_predictions || !Array.isArray(results.top_predictions)) {
        showError('Error: Formato de respuesta de la API inválido');
        return; // termina ejecucion if data son invalidos
    }

    // crea el contenedor principal for los resultados
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card'; // aplica estilos css

    // crea seccion de encabezado with resultado principal
    const header = document.createElement('div');
    header.className = 'result-header';
    
    // elemento for mostrar la breed mas probable
    const title = document.createElement('h2');
    title.className = 'result-title';
    
    // obtiene la breed principal de la respuesta with fallback seguro
    const mainBreed = results.recommendation?.most_likely || results.top_predictions[0]?.breed || 'Desconocido';
    title.textContent = `🐕 ${formatBreedName(mainBreed)}`;
    
    // crea badge de confianza with color segun nivel
    const confidenceBadge = document.createElement('div');
    
    // obtiene confianza principal with fallback seguro
    const mainConfidence = results.recommendation?.confidence || results.top_predictions[0]?.confidence || 0;
    confidenceBadge.className = `confidence-badge ${getConfidenceLevel(mainConfidence)}`;
    confidenceBadge.textContent = `${(mainConfidence * 100).toFixed(1)}% confianza`;
    
    // ensambla el encabezado
    header.appendChild(title);
    header.appendChild(confidenceBadge);

    // crea list de las best predictions
    const predictionsList = document.createElement('div');
    predictionsList.className = 'predictions-list';

    // muestra only las 5 best predictions for no saturar interfaz
    const topPredictions = results.top_predictions.slice(0, 5);
    topPredictions.forEach((prediction, index) => {
        const item = createPredictionItem(prediction, index + 1); // index + 1 for ranking humano
        predictionsList.appendChild(item);
    });

    // ensambla toda la tarjeta de resultados
    resultCard.appendChild(header);
    resultCard.appendChild(predictionsList);
    container.appendChild(resultCard);

    showSection('resultsContainer'); // hace visible la seccion de resultados
}

// crea un elemento visual individual for cada prediction de breed
// incluye ranking, name, barra de confianza y porcentaje
function createPredictionItem(prediction, rank) {
    // contenedor principal of the item with estilo especial for el primero
    const item = document.createElement('div');
    item.className = `prediction-item ${rank === 1 ? 'top' : ''}`; // class especial for #1

    // seccion izquierda with ranking y name de breed
    const breedInfo = document.createElement('div');
    breedInfo.className = 'breed-info';

    // badge circular with numero de ranking
    const rankBadge = document.createElement('div');
    rankBadge.className = 'breed-rank';
    rankBadge.textContent = rank; // muestra 1, 2, 3, etc

    // name de la breed formateado for lectura humana
    const breedName = document.createElement('div');
    breedName.className = 'breed-name';
    breedName.textContent = formatBreedName(prediction.breed);

    // ensambla seccion de informacion de breed
    breedInfo.appendChild(rankBadge);
    breedInfo.appendChild(breedName);

    // seccion derecha with visualizacion de confianza
    const confidenceInfo = document.createElement('div');
    confidenceInfo.className = 'confidence-info';

    // contenedor de barra de progreso
    const confidenceBar = document.createElement('div');
    confidenceBar.className = 'confidence-bar';

    // barra de progreso that se llena segun porcentaje de confianza
    const confidenceFill = document.createElement('div');
    confidenceFill.className = 'confidence-fill';
    confidenceFill.style.width = `${prediction.confidence * 100}%`; // ancho proporcional

    confidenceBar.appendChild(confidenceFill);

    // texto with porcentaje numerico
    const confidencePercent = document.createElement('div');
    confidencePercent.className = 'confidence-percent';
    confidencePercent.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    // ensambla seccion de confianza
    confidenceInfo.appendChild(confidenceBar);
    confidenceInfo.appendChild(confidencePercent);

    // ensambla item completo
    item.appendChild(breedInfo);
    item.appendChild(confidenceInfo);

    return item; // devuelve elemento listo for insertar en dom
}

// convierte names tecnicos de breeds en formato legible for humanos
// transforma 'golden_retriever' en 'Golden Retriever'
function formatBreedName(breedName) {
    return breedName
        .split('_')                                    // separa por guiones bajos
        .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // capitaliza cada palabra
        .join(' ');                                    // une with espacios
}

// determina el nivel de confianza for aplicar colores apropiados
// ayuda to the user a interpretar rapidamente that tan seguro this el model
function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'high';    // verde for alta confianza
    if (confidence >= 0.5) return 'medium';  // amarillo for confianza media
    return 'low';                             // rojo for baja confianza
}

// reinicia toda la interfaz to the estado inicial for nueva classification
// limpia data anteriores y restaura vista de upload
function resetInterface() {
    state.currentImage = null;  // borra referencia to the file anterior
    state.results = null;       // borra resultados anteriores
    
    elements.fileInput.value = '';    // limpia input de file
    elements.previewImg.src = '';     // borra image mostrada
    
    // controla visibilidad de secciones
    hideSection('imagePreview');      // oculta preview de image
    hideSection('loadingContainer');  // oculta indicador de load
    hideSection('resultsContainer');  // oculta resultados anteriores
    showSection('uploadArea');        // muestra zona de upload original
}

// hace visible una seccion especifica de la interfaz
// centraliza el control de visibilidad for consistency
function showSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'block'; // hace visible with display block
    }
}

// oculta una seccion especifica de la interfaz
// centraliza el control de visibilidad for consistency
function hideSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'none'; // oculta with display none
    }
}

// Implementation note.
// crea elementos temporales that aparecen y desaparecen automaticamente
function showError(message) {
    // crea elemento de notificacion toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    
    // aplica estilos css inline for posicionamiento y apariencia
    toast.style.cssText = `
        position: fixed;           /* posicion fija en viewport */
        top: 20px;                /* 20px desde arriba */
        right: 20px;              /* 20px desde derecha */
        background: #ef4444;      /* Implementation note. */
        color: white;             /* texto blanco */
        padding: 1rem 1.5rem;     /* espaciado interno */
        border-radius: 12px;      /* bordes redondeados */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); /* sombra */
        z-index: 1000;            /* por encima de otros elementos */
        animation: slideInRight 0.3s ease-out; /* animacion de entrada */
        max-width: 400px;         /* ancho maximo */
        font-weight: 500;         /* peso de fuente */
    `;
    toast.textContent = message; // contenido of the mensaje

    // crea estilos de animacion dinamicamente
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%); /* empieza fuera de pantalla */
                opacity: 0;
            }
            to {
                transform: translateX(0);    /* termina en posicion normal */
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style); // agrega estilos to the documento

    document.body.appendChild(toast); // agrega toast to the documento

    // remueve automaticamente el toast despues de 5 segundos
    setTimeout(() => {
        toast.remove();  // elimina toast of the dom
        style.remove();  // elimina estilos of the dom
    }, 5000);
}

// verifica that la API of the model this disponible y respondiendo
// realiza un health check to the load la pagina for detectar problemas temprano
async function checkAPIConnection() {
    try {
        // peticion simple to the endpoint de salud de la API
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            console.log('✅ Conexión con API establecida'); // confirmacion en consola
        } else {
            throw new Error('API no disponible'); // Implementation note.
        }
    } catch (error) {
        // informa to the user if hay problemas de conectividad
        console.warn('⚠️ No se pudo conectar con la API:', error.message);
        showError('No se pudo conectar con el servidor. Asegúrate de que la API esté ejecutándose en el puerto 8000');
    }
}

// inicializa el system de particulas animadas en el fondo
// crea un efecto visual atractivo without interferir with la funcionalidad
function initParticles() {
    const canvas = document.getElementById('particles');   // obtiene canvas for dibujo
    const ctx = canvas.getContext('2d');                  // contexto 2d for dibujar
    
    // Implementation note.
    function resizeCanvas() {
        canvas.width = window.innerWidth;   // ancho igual to the viewport
        canvas.height = window.innerHeight; // alto igual to the viewport
    }
    
    resizeCanvas(); // Implementation note.
    window.addEventListener('resize', resizeCanvas); // reajusta to the cambiar ventana

    // configuration of the system de particulas
    const particles = [];           // array that contiene all las particulas
    const particleCount = 50;       // numero total de particulas

    // crea particulas iniciales with propiedades aleatorias
    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,          // posicion x aleatoria
            y: Math.random() * canvas.height,         // posicion y aleatoria
            size: Math.random() * 3 + 1,              // Implementation note.
            speedX: (Math.random() - 0.5) * 0.5,      // velocidad horizontal
            speedY: (Math.random() - 0.5) * 0.5,      // velocidad vertical
            opacity: Math.random() * 0.5 + 0.2        // opacidad entre 0.2 y 0.7
        });
    }

    // function de animacion that se ejecuta continuamente
    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height); // limpia frame anterior

        // actualiza y dibuja cada particula
        particles.forEach(particle => {
            // actualiza posicion basada en velocidad
            particle.x += particle.speedX;
            particle.y += particle.speedY;

            // rebote en bordes de pantalla
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

    animateParticles(); // inicia loop de animacion
}

// convierte bytes a formato legible for humanos
// Implementation note.
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';  // caso especial for files vacios
    
    const k = 1024;                     // factor de conversion
    const sizes = ['Bytes', 'KB', 'MB', 'GB']; // unidades disponibles
    
    // Implementation note.
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    // formatea numero with 2 decimales y agrega unidad apropiada
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// obtiene informacion detallada of the model desde la API
// util for debugging y verificacion de configuration
async function getModelInfo() {
    try {
        // peticion to the endpoint de informacion of the model
        const response = await fetch(`${API_BASE_URL}/model-info`);
        
        if (response.ok) {
            const info = await response.json();
            console.log('📊 Información del modelo:', info); // muestra en consola
            return info; // devuelve data for uso posterior
        }
    } catch (error) {
        console.warn('No se pudo obtener información del modelo:', error);
    }
    return null; // Implementation note.
}