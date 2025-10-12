# VeNTUre Backend - FastAPI Food Analysis API

A comprehensive FastAPI backend for food logging and nutrition analysis using AI/ML models for food classification, segmentation, and nutritional estimation.

## 🏗️ Project Structure

```
backend/
├── .env                    # Environment variables (Supabase keys, etc.)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── venv/                  # Virtual environment (optional)
└── app/
    ├── __init__.py        # Makes app a Python package
    ├── main.py            # FastAPI application entry point
    ├── main2.py           # Alternative/backup main file
    │
    ├── routes/            # API route definitions
    │   ├── __init__.py
    │   ├── auth.py        # Authentication endpoints
    │   ├── food.py        # Food logging endpoints
    │   ├── inference.py   # AI/ML inference endpoints
    │   └── metrics.py     # Analytics and metrics endpoints
    │
    ├── models/            # Pydantic data models
    │   ├── __init__.py
    │   └── schemas.py     # Request/response schemas
    │
    ├── db/                # Database integration
    │   ├── __init__.py
    │   └── supabase_client.py  # Supabase client configuration
    │
    ├── ml/                # Machine Learning utilities
    │   ├── __init__.py
    │   ├── class_names.py      # Food classification labels
    │   ├── inference.py        # ML inference logic
    │   ├── inference_service.py # Service layer for ML
    │   ├── preprocess.py       # Image preprocessing
    │   └── vit_inference.py    # Vision Transformer inference
    │
    ├── food_ai/           # Advanced AI models and processing
    │   ├── segmentation_wrapper.py  # Food segmentation model wrapper
    │   ├── models/        # AI model files and configurations
    │   │   ├── model_config.py     # Model configuration
    │   │   ├── classification/     # Food classification models
    │   │   ├── segmentation/       # Food segmentation models
    │   │   ├── utensil_detection/  # Utensil detection models
    │   │   └── volume/            # Volume estimation models
    │   └── storage/       # Data storage utilities
    │
    └── ai_models/         # Pre-trained model files
        └── vit_small.onnx # Vision Transformer model
```

## 🚀 Key Features

### 1. Food Analysis Pipeline
1. Whole food image gets classified by ViT to indentify food class.
2. Image is segmented by segmentation model (Segformer)
3. Segmented images are classified by ViT again 
4. Compare the confidence threshold between whole image vs segmented parts

Segmented images gives higher threshold:
- Segmented parts with threshold > 0.4 are considered -> consider the ingredients else we use broad food categories

Whole image gives higher threshold:
- Volume of segments are combined.
5. Utensil gets detected by YOLO model 
6. Depth model (Depth Anything) incorportates the utensil to estimate volume of (whole/segmented)
7. Mass is calculated when using volume x density (obtained from usda)
8. Calculate calories through food mapping (Nutrionix_api -> 233/244) 




## 2. API Endpoints

#### Core Analysis
- `POST /analyze` - Complete food image analysis with nutrition calculation
- `GET /health` - Health check endpoint
- `GET /` - Root endpoint

#### Food Management
- Food logging and retrieval
- Meal categorization (breakfast, lunch, dinner, snacks)
- User-specific food history

#### Authentication
- User authentication and authorization
- Profile management

#### Metrics & Analytics
- User nutrition tracking
- Meal statistics
- Progress analytics

## 🛠️ Technology Stack

### Core Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI web server
- **Pydantic**: Data validation using Python type annotations

### Database & Storage
- **Supabase**: Backend-as-a-Service (BaaS) for database and storage
- **PostgreSQL**: Relational database (via Supabase)
- **Supabase Storage**: File storage for meal images

### AI/ML Stack
- **ONNX Runtime**: Cross-platform ML inference
- **Vision Transformer (ViT)**: Food classification
- **SegFormer**: Food segmentation
- **YOLO**: Utensil detection for scale reference
- **Depth Estimation**: Volume calculation using depth maps

### Image Processing
- **PIL (Pillow)**: Image manipulation
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical operations
- **scikit-image**: Image processing utilities

## 🔧 Setup & Installation

### Prerequisites
- Python 3.11+
- Conda (recommended) or pip
- Supabase account and project

### Installation Steps

1. **Clone and navigate to backend**
   ```bash
   cd backend
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n venture python=3.11
   conda activate venture
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file with:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   SUPABASE_ANON_KEY=your_anon_key
   ```

5. **Download AI Models**
   - Place your ONNX models in the appropriate `food_ai/models/` subdirectories
   - Ensure `vit_small.onnx` is in `ai_models/`

### Running the Server

```bash
# Make sure you're in the backend directory
cd backend

# Activate conda environment
conda activate venture

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080`

## 📋 API Usage

### Complete Food Analysis
```bash
POST /analyze
Content-Type: application/json

{
  "bucket": "meal_images",
  "path": "user123/1234567890.jpg",
  "image_base64": "base64_encoded_image_data",
  "confidence_threshold": 0.3,
  "enable_volume": true,
  "enable_nutrition": true
}
```

### Response Structure
```json
{
  "status": "success",
  "classification": {
    "top_prediction": {
      "class_name": "apple",
      "confidence": 0.95
    }
  },
  "segmentation": {
    "detections": [...]
  },
  "volume_estimates": {
    "total_volume_ml": 150.0
  },
  "nutrition": {
    "summary": {
      "total_calories": 78,
      "total_protein_g": 0.3,
      "total_carbs_g": 20.6,
      "total_fat_g": 0.2
    }
  }
}
```

## 🔒 Security Considerations

- Environment variables for sensitive data
- Supabase Row Level Security (RLS) for data access control
- CORS configuration for frontend integration
- Input validation using Pydantic models

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'app'**
   - Ensure you're running uvicorn from the `backend` directory
   - Check that `app/__init__.py` exists

2. **Model loading errors**
   - Verify ONNX model files are in correct locations
   - Check model file permissions and paths

3. **Supabase connection errors**
   - Verify `.env` file configuration
   - Check Supabase project status and API keys

4. **CORS errors**
   - Update CORS settings in `main.py` for your frontend domain

## 📈 Performance Considerations

- **Model Optimization**: ONNX models for faster inference
- **Async Processing**: FastAPI's async capabilities for concurrent requests
- **Image Processing**: Efficient preprocessing pipelines
- **Caching**: Consider implementing Redis for model caching
- **Rate Limiting**: Implement for production use

## 🔄 Development Workflow

1. **Code Changes**: Make changes to relevant modules
2. **Testing**: Test endpoints using FastAPI's interactive docs at `/docs`
3. **Debugging**: Use logging throughout the application
4. **Deployment**: Use Docker or cloud platforms for production

## 📝 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: `http://localhost:8080/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8080/redoc` (ReDoc)

## 🤝 Contributing

1. Follow Python PEP 8 style guidelines
2. Add type hints to all functions
3. Update this README when adding new features
4. Test all endpoints before committing

## 📞 Support

For issues related to:
- **FastAPI**: Check the official FastAPI documentation
- **Supabase**: Refer to Supabase documentation
- **ML Models**: Ensure proper ONNX model compatibility
