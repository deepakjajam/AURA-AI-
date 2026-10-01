from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.core.database import init_db
from app.routers import auth, aura, chat, keys, training

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield

app = FastAPI(title='AURA AI API', version='0.1.0', lifespan=lifespan)
app.include_router(auth.router, prefix='/api/auth', tags=['authentication'])
app.include_router(aura.router, prefix='/api/aura', tags=['AURA Online'])
app.include_router(keys.router, prefix='/api/keys', tags=['API keys'])
app.include_router(chat.router, prefix='/api/chat', tags=['inference'])
app.include_router(training.router, prefix='/api/training', tags=['training'])
app.mount('/static', StaticFiles(directory='app/static'), name='static')

@app.get('/', response_class=HTMLResponse, include_in_schema=False)
def home():
    with open('app/static/index.html', encoding='utf-8') as page:
        return page.read()

@app.get('/health', tags=['system'])
def health():
    return {'status': 'ok', 'service': 'aura-api'}
