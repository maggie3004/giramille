# GenVec Frontend

Next.js + Tailwind frontend that replicates the exact UI design using provided static images.

## Features

- **Prompt Box**: Text input with scroll, state-backed
- **Generate Buttons**: 
  - GERAR VETOR: Downloads SVG file
  - GERAR PNG: Creates placeholder image and adds to history
- **History Panel**: Scrollable thumbnails, click to preview
- **Options**: Estilo (Style), Cores (Colors), Proporção (Aspect Ratio)
- **Preview**: Shows generated/selected images

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Backend Integration

Replace the simulation functions in `app/page.tsx`:

- `handleGeneratePNG()` → call your AI backend
- `handleGenerateVector()` → call your vectorization backend

The state management is already set up to handle real responses.

## File Structure

```
frontend/
├── app/
│   ├── page.tsx          # Main UI with pixel-perfect positioning
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Tailwind styles
├── components/
│   └── History.tsx       # History panel component
└── public/static/        # UI images (copy from parent /static)
```

## Customization

- Adjust absolute positioning in `page.tsx` if hitboxes don't align
- Add more style/color options in the selectors
- Extend history with metadata (prompt, timestamp, etc.)
