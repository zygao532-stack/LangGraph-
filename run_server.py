import sys
sys.path.insert(0, r"D:\gongjutai\langgraph-my-project")

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8001, reload=False)
