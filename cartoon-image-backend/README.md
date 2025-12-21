# Cartoon Image Backend

This project is designed to generate cartoon-style images based on user input. It serves as a backend service that processes requests for image generation and retrieval.

## Project Structure

- **src/**: Contains the main application code.
  - **index.ts**: Entry point of the application.
  - **server.ts**: Server configuration and middleware setup.
  - **controllers/**: Contains the `imageController.ts` for handling image-related requests.
  - **services/**: Contains the `imageService.ts` for business logic related to image generation.
  - **models/**: Contains the `imageModel.ts` which defines the structure of image data and includes methods for generating cartoon-style images.
  - **utils/**: Contains utility functions for preprocessing input data.
  - **routes/**: Contains the `imageRoutes.ts` which connects controller methods to specific endpoints.
  
- **tests/**: Contains unit tests for the application.
  - **imageModel.test.ts**: Tests for the `ImageModel` class.

- **package.json**: Configuration file for npm, listing dependencies and scripts.
- **tsconfig.json**: TypeScript configuration file specifying compiler options.
- **.env.example**: Example of environment variables needed for the application.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd cartoon-image-backend
   ```
3. Install the dependencies:
   ```
   npm install
   ```

## Usage

To start the server, run:
```
npm start
```

The server will be running on the specified port in `server.ts`. You can send requests to the image generation endpoints defined in `imageRoutes.ts`.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.