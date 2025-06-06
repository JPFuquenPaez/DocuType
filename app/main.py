"""Main application."""

import uvicorn

if __name__ == "__main__":
    # Start the Uvicorn server programmatically
    uvicorn.run("app.webservice:app", port=9000, host="0.0.0.0", reload=True)
