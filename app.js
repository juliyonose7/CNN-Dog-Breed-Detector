// url base donde se ejecuta la api del modelo de clasificacion
// localhost puerto 8000 es donde fastapi sirve el modelo resnet50
const API_BASE_URL = 'http://localhost:8000';

// objeto que mantiene el estado global de la aplicacion
// centraliza toda la informacion importante durante el uso
const state = {
    currentImage: null,     // archivo de imagen actualmente seleccionado
    isLoading: false,      // indica si hay una operacion en progreso
    results: null          // almacena los resultados de la ultima clasificacion
};

// referencias a elementos del dom para manipulacion eficiente
// evita buscar elementos repetidamente con getelementbyid
const elements = {
    uploadArea: null,        // zona donde se arrastra o selecciona imagenes
    fileInput: null,         // input oculto que maneja seleccion de archivos
    imagePreview: null,      // contenedor que muestra preview de imagen
    previewImg: null,        // elemento img que renderiza la imagen subida
    resetBtn: null,          // boton para limpiar y subir nueva imagen
    loadingContainer: null,  // indicador visual de procesamiento
    resultsContainer: null   // area donde se muestran resultados de clasificacion
};

// evento que se ejecuta automaticamente cuando el dom esta completamente cargado
// garantiza que todos los elementos html esten disponibles antes de manipularlos
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();    // obtiene referencias a todos los elementos necesarios
    setupEventListeners();   // configura todos los manejadores de eventos
    initParticles();        // inicia la animacion de particulas en el fondo
    checkAPIConnection();    // verifica que la api del modelo este disponible
});

// obtiene referencias a todos los elementos html importantes y los guarda
// esto optimiza el rendimiento evitando busquedas repetidas en el dom
function initializeElements() {
    elements.uploadArea = document.getElementById('uploadArea');           // zona de drag and drop
    elements.fileInput = document.getElementById('fileInput');             // input para seleccionar archivos
    elements.imagePreview = document.getElementById('imagePreview');       // contenedor de vista previa
    elements.previewImg = document.getElementById('previewImg');           // imagen mostrada al usuario
    elements.resetBtn = document.getElementById('resetBtn');               // boton de reinicio
    elements.loadingContainer = document.getElementById('loadingContainer'); // indicador de carga
    elements.resultsContainer = document.getElementById('resultsContainer'); // area de resultados
}

// configura todos los manejadores de eventos para interaccion del usuario
// establece como la interfaz responde a clics, drag and drop, etc
function setupEventListeners() {
    // evento de clic en zona de upload activa el selector de archivos
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    
    // eventos de drag and drop para arrastrar imagenes desde el explorador
    elements.uploadArea.addEventListener('dragover', handleDragOver);   // cuando se arrastra sobre area
    elements.uploadArea.addEventListener('dragleave', handleDragLeave); // cuando se sale del area
    elements.uploadArea.addEventListener('drop', handleDrop);           // cuando se suelta archivo

    // evento cuando usuario selecciona archivo manualmente
    elements.fileInput.addEventListener('change', handleFileSelect);

    // evento del boton para resetear y subir nueva imagen
    elements.resetBtn.addEventListener('click', resetInterface);

    // previene comportamiento por defecto del navegador en drag and drop
    // necesario para que funcione correctamente la funcionalidad personalizada
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, preventDefaults);
        document.body.addEventListener(eventName, preventDefaults);
    });
}

// previene las acciones por defecto del navegador para eventos de arrastre
// necesario para implementar drag and drop personalizado
function preventDefaults(e) {
    e.preventDefault();  // cancela la accion por defecto del navegador
    e.stopPropagation(); // evita que el evento se propague a elementos padre
}

// maneja cuando un archivo se arrastra sobre la zona de upload
// cambia el estilo visual para indicar zona activa
function handleDragOver(e) {
    elements.uploadArea.classList.add('dragover'); // aplica estilo visual de zona activa
}

// maneja cuando el archivo sale de la zona de upload
// remueve el estilo visual de zona activa
function handleDragLeave(e) {
    elements.uploadArea.classList.remove('dragover'); // quita estilo de zona activa
}

// maneja cuando se suelta un archivo en la zona de upload
// procesa el primer archivo arrastrado
function handleDrop(e) {
    elements.uploadArea.classList.remove('dragover'); // quita estilo visual
    const files = e.dataTransfer.files;               // obtiene archivos arrastrados
    if (files.length > 0) {
        processFile(files[0]); // procesa solo el primer archivo
    }
}

// maneja cuando usuario selecciona archivo con el boton
// procesa el archivo seleccionado del input
function handleFileSelect(e) {
    const file = e.target.files[0]; // obtiene primer archivo seleccionado
    if (file) {
        processFile(file); // procesa el archivo
    }
}

// procesa y valida el archivo seleccionado antes de enviarlo a clasificacion
// realiza verificaciones de seguridad y formato antes de continuar
function processFile(file) {
    // verifica que el archivo sea realmente una imagen
    // startswith verifica el tipo mime del archivo
    if (!file.type.startsWith('image/')) {
        showError('Por favor selecciona un archivo de imagen válido');
        return; // termina ejecucion si no es imagen
    }

    // verifica que el tamaño no exceda 10mb para evitar problemas de memoria
    // 10 * 1024 * 1024 = 10485760 bytes = 10mb
    if (file.size > 10 * 1024 * 1024) {
        showError('La imagen es demasiado grande. Máximo 10MB');
        return; // termina ejecucion si es muy grande
    }

    state.currentImage = file;    // guarda referencia al archivo en estado global
    displayImagePreview(file);    // muestra preview de la imagen al usuario
    classifyImage(file);         // inicia proceso de clasificacion con api
}

// muestra una vista previa de la imagen seleccionada antes de procesarla
// convierte el archivo en una url que puede mostrar el navegador
function displayImagePreview(file) {
    const reader = new FileReader(); // api del navegador para leer archivos
    
    // evento que se ejecuta cuando filereader termina de leer el archivo
    reader.onload = function(e) {
        elements.previewImg.src = e.target.result; // asigna datos de imagen al elemento img
        showSection('imagePreview');               // hace visible la seccion de preview
        hideSection('uploadArea');                 // oculta la zona de upload original
    };
    
    // inicia la lectura del archivo como data url base64
    // esto convierte la imagen en texto que el navegador puede mostrar
    reader.readAsDataURL(file);
}

// envia la imagen a la api del modelo para clasificacion de raza
// maneja todo el proceso de comunicacion con el backend
async function classifyImage(file) {
    try {
        showSection('loadingContainer');  // muestra indicador de carga al usuario
        hideSection('resultsContainer');  // oculta resultados anteriores si existen
        state.isLoading = true;          // marca estado global como procesando

        // prepara datos para envio multipart form data requerido por fastapi
        const formData = new FormData();
        formData.append('file', file);   // adjunta archivo con nombre 'file'

        console.log('🔄 Enviando imagen a la API...'); // log para debugging
        
        // realiza peticion http post al endpoint de prediccion
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData  // no se especifica content-type, fetch lo hace automaticamente
        });

        // verifica si la respuesta http fue exitosa
        if (!response.ok) {
            const errorText = await response.text(); // obtiene mensaje de error detallado
            console.error('Error de la API:', errorText);
            throw new Error(`Error del servidor: ${response.status} - ${errorText}`);
        }

        // convierte respuesta json en objeto javascript
        const results = await response.json();
        console.log('✅ Respuesta de la API:', results); // log para debugging
        
        // verifica que la api devolvio una respuesta valida
        if (!results || !results.success) {
            throw new Error('La API devolvió una respuesta de error');
        }

        state.results = results;  // guarda resultados en estado global
        displayResults(results);  // muestra resultados al usuario

    } catch (error) {
        console.error('❌ Error en clasificación:', error);
        showError(`Error al clasificar la imagen: ${error.message}`);
    } finally {
        state.isLoading = false;         // marca que proceso termino
        hideSection('loadingContainer'); // oculta indicador de carga
    }
}

// construye y muestra la interfaz de resultados con las predicciones del modelo
// crea dinamicamente elementos html para una presentacion visual atractiva
function displayResults(results) {
    const container = elements.resultsContainer;
    
    container.innerHTML = ''; // limpia cualquier resultado anterior

    // verifica que los datos de la api tengan el formato correcto
    // evita errores si la respuesta esta mal formada
    if (!results || !results.top_predictions || !Array.isArray(results.top_predictions)) {
        showError('Error: Formato de respuesta de la API inválido');
        return; // termina ejecucion si datos son invalidos
    }

    // crea el contenedor principal para los resultados
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card'; // aplica estilos css

    // crea seccion de encabezado con resultado principal
    const header = document.createElement('div');
    header.className = 'result-header';
    
    // elemento para mostrar la raza mas probable
    const title = document.createElement('h2');
    title.className = 'result-title';
    
    // obtiene la raza principal de la respuesta con fallback seguro
    const mainBreed = results.recommendation?.most_likely || results.top_predictions[0]?.breed || 'Desconocido';
    title.textContent = `🐕 ${formatBreedName(mainBreed)}`;
    
    // crea badge de confianza con color segun nivel
    const confidenceBadge = document.createElement('div');
    
    // obtiene confianza principal con fallback seguro
    const mainConfidence = results.recommendation?.confidence || results.top_predictions[0]?.confidence || 0;
    confidenceBadge.className = `confidence-badge ${getConfidenceLevel(mainConfidence)}`;
    confidenceBadge.textContent = `${(mainConfidence * 100).toFixed(1)}% confianza`;
    
    // ensambla el encabezado
    header.appendChild(title);
    header.appendChild(confidenceBadge);

    // crea lista de las mejores predicciones
    const predictionsList = document.createElement('div');
    predictionsList.className = 'predictions-list';

    // muestra solo las 5 mejores predicciones para no saturar interfaz
    const topPredictions = results.top_predictions.slice(0, 5);
    topPredictions.forEach((prediction, index) => {
        const item = createPredictionItem(prediction, index + 1); // index + 1 para ranking humano
        predictionsList.appendChild(item);
    });

    // ensambla toda la tarjeta de resultados
    resultCard.appendChild(header);
    resultCard.appendChild(predictionsList);
    container.appendChild(resultCard);

    showSection('resultsContainer'); // hace visible la seccion de resultados
}

// crea un elemento visual individual para cada prediccion de raza
// incluye ranking, nombre, barra de confianza y porcentaje
function createPredictionItem(prediction, rank) {
    // contenedor principal del item con estilo especial para el primero
    const item = document.createElement('div');
    item.className = `prediction-item ${rank === 1 ? 'top' : ''}`; // clase especial para #1

    // seccion izquierda con ranking y nombre de raza
    const breedInfo = document.createElement('div');
    breedInfo.className = 'breed-info';

    // badge circular con numero de ranking
    const rankBadge = document.createElement('div');
    rankBadge.className = 'breed-rank';
    rankBadge.textContent = rank; // muestra 1, 2, 3, etc

    // nombre de la raza formateado para lectura humana
    const breedName = document.createElement('div');
    breedName.className = 'breed-name';
    breedName.textContent = formatBreedName(prediction.breed);

    // ensambla seccion de informacion de raza
    breedInfo.appendChild(rankBadge);
    breedInfo.appendChild(breedName);

    // seccion derecha con visualizacion de confianza
    const confidenceInfo = document.createElement('div');
    confidenceInfo.className = 'confidence-info';

    // contenedor de barra de progreso
    const confidenceBar = document.createElement('div');
    confidenceBar.className = 'confidence-bar';

    // barra de progreso que se llena segun porcentaje de confianza
    const confidenceFill = document.createElement('div');
    confidenceFill.className = 'confidence-fill';
    confidenceFill.style.width = `${prediction.confidence * 100}%`; // ancho proporcional

    confidenceBar.appendChild(confidenceFill);

    // texto con porcentaje numerico
    const confidencePercent = document.createElement('div');
    confidencePercent.className = 'confidence-percent';
    confidencePercent.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    // ensambla seccion de confianza
    confidenceInfo.appendChild(confidenceBar);
    confidenceInfo.appendChild(confidencePercent);

    // ensambla item completo
    item.appendChild(breedInfo);
    item.appendChild(confidenceInfo);

    return item; // devuelve elemento listo para insertar en dom
}

// convierte nombres tecnicos de razas en formato legible para humanos
// transforma 'golden_retriever' en 'Golden Retriever'
function formatBreedName(breedName) {
    return breedName
        .split('_')                                    // separa por guiones bajos
        .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // capitaliza cada palabra
        .join(' ');                                    // une con espacios
}

// determina el nivel de confianza para aplicar colores apropiados
// ayuda al usuario a interpretar rapidamente que tan seguro esta el modelo
function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'high';    // verde para alta confianza
    if (confidence >= 0.5) return 'medium';  // amarillo para confianza media
    return 'low';                             // rojo para baja confianza
}

// reinicia toda la interfaz al estado inicial para nueva clasificacion
// limpia datos anteriores y restaura vista de upload
function resetInterface() {
    state.currentImage = null;  // borra referencia al archivo anterior
    state.results = null;       // borra resultados anteriores
    
    elements.fileInput.value = '';    // limpia input de archivo
    elements.previewImg.src = '';     // borra imagen mostrada
    
    // controla visibilidad de secciones
    hideSection('imagePreview');      // oculta preview de imagen
    hideSection('loadingContainer');  // oculta indicador de carga
    hideSection('resultsContainer');  // oculta resultados anteriores
    showSection('uploadArea');        // muestra zona de upload original
}

// hace visible una seccion especifica de la interfaz
// centraliza el control de visibilidad para consistency
function showSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'block'; // hace visible con display block
    }
}

// oculta una seccion especifica de la interfaz
// centraliza el control de visibilidad para consistency
function hideSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'none'; // oculta con display none
    }
}

// muestra mensajes de error al usuario mediante notificaciones toast
// crea elementos temporales que aparecen y desaparecen automaticamente
function showError(message) {
    // crea elemento de notificacion toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    
    // aplica estilos css inline para posicionamiento y apariencia
    toast.style.cssText = `
        position: fixed;           /* posicion fija en viewport */
        top: 20px;                /* 20px desde arriba */
        right: 20px;              /* 20px desde derecha */
        background: #ef4444;      /* fondo rojo para error */
        color: white;             /* texto blanco */
        padding: 1rem 1.5rem;     /* espaciado interno */
        border-radius: 12px;      /* bordes redondeados */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); /* sombra */
        z-index: 1000;            /* por encima de otros elementos */
        animation: slideInRight 0.3s ease-out; /* animacion de entrada */
        max-width: 400px;         /* ancho maximo */
        font-weight: 500;         /* peso de fuente */
    `;
    toast.textContent = message; // contenido del mensaje

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
    document.head.appendChild(style); // agrega estilos al documento

    document.body.appendChild(toast); // agrega toast al documento

    // remueve automaticamente el toast despues de 5 segundos
    setTimeout(() => {
        toast.remove();  // elimina toast del dom
        style.remove();  // elimina estilos del dom
    }, 5000);
}

// verifica que la api del modelo este disponible y respondiendo
// realiza un health check al cargar la pagina para detectar problemas temprano
async function checkAPIConnection() {
    try {
        // peticion simple al endpoint de salud de la api
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            console.log('✅ Conexión con API establecida'); // confirmacion en consola
        } else {
            throw new Error('API no disponible'); // error si respuesta no es 200
        }
    } catch (error) {
        // informa al usuario si hay problemas de conectividad
        console.warn('⚠️ No se pudo conectar con la API:', error.message);
        showError('No se pudo conectar con el servidor. Asegúrate de que la API esté ejecutándose en el puerto 8000');
    }
}

// inicializa el sistema de particulas animadas en el fondo
// crea un efecto visual atractivo sin interferir con la funcionalidad
function initParticles() {
    const canvas = document.getElementById('particles');   // obtiene canvas para dibujo
    const ctx = canvas.getContext('2d');                  // contexto 2d para dibujar
    
    // ajusta el canvas al tamaño completo de la ventana
    function resizeCanvas() {
        canvas.width = window.innerWidth;   // ancho igual al viewport
        canvas.height = window.innerHeight; // alto igual al viewport
    }
    
    resizeCanvas(); // ajusta tamaño inicial
    window.addEventListener('resize', resizeCanvas); // reajusta al cambiar ventana

    // configuracion del sistema de particulas
    const particles = [];           // array que contiene todas las particulas
    const particleCount = 50;       // numero total de particulas

    // crea particulas iniciales con propiedades aleatorias
    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,          // posicion x aleatoria
            y: Math.random() * canvas.height,         // posicion y aleatoria
            size: Math.random() * 3 + 1,              // tamaño entre 1 y 4 pixeles
            speedX: (Math.random() - 0.5) * 0.5,      // velocidad horizontal
            speedY: (Math.random() - 0.5) * 0.5,      // velocidad vertical
            opacity: Math.random() * 0.5 + 0.2        // opacidad entre 0.2 y 0.7
        });
    }

    // funcion de animacion que se ejecuta continuamente
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

// convierte bytes a formato legible para humanos
// util para mostrar tamaños de archivo de manera comprensible
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';  // caso especial para archivos vacios
    
    const k = 1024;                     // factor de conversion
    const sizes = ['Bytes', 'KB', 'MB', 'GB']; // unidades disponibles
    
    // calcula que unidad usar basado en tamaño
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    // formatea numero con 2 decimales y agrega unidad apropiada
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// obtiene informacion detallada del modelo desde la api
// util para debugging y verificacion de configuracion
async function getModelInfo() {
    try {
        // peticion al endpoint de informacion del modelo
        const response = await fetch(`${API_BASE_URL}/model-info`);
        
        if (response.ok) {
            const info = await response.json();
            console.log('📊 Información del modelo:', info); // muestra en consola
            return info; // devuelve datos para uso posterior
        }
    } catch (error) {
        console.warn('No se pudo obtener información del modelo:', error);
    }
    return null; // devuelve null si hay error
}