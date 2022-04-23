#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <map>
#include <string>


#include <freetype2/ft2build.h>
#include FT_FREETYPE_H

#include "common/fileSystem.hpp"
#include "common/shader_s.hpp"

static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
static void processInput(GLFWwindow* window);
static void renderText(Shader &shader, std::string text, float x, float y, float scale, glm::vec3 color);

//settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//hold all state information relevant to a character as loaded using freetype
struct Character {
    GLuint textureId; // Id handle of the glyph texture
    glm::ivec2 size; //size of glyph
    glm::ivec2 bearing; //offset from baseline to left/top of glyph
    unsigned int advance; //horizontal offset to advance to next glyph
};

std::map<GLchar, Character> Characters;
GLuint VAO, VBO;

int main(int argc, char *argv[])
{
    GLFWwindow *window;
    //glfw initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    //glfw window creation
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    //glad : load all OpenGL function pointer
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //OpenGL state
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //compile and setup the shader
    Shader shader("shaders/text.vs", "shaders/text.fs");
    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(SCR_WIDTH), 0.0f, static_cast<float>(SCR_HEIGHT));
    shader.use();
    glUniformMatrix4fv(glGetUniformLocation(shader.Id, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    /* freetype load glyph */
    FT_Library ft;
    /** All functions return a value different than 0 whenever an error occurred **/
    if (FT_Init_FreeType(&ft)){
        std::cerr << "Error::FreeType: Failed to load font_name" << std::endl;
        return -1;
    }

    //find path to font
    std::string fontName = fileSystem::getResource("../resources/fonts/Antonio-Bold.ttf");
    if (fontName.empty()) {
        std::cerr << "Error::FreeType: Failed to load font" << std::endl;
        return -1;
    }

    //load font as face
    FT_Face face;
    if (FT_New_Face(ft, fontName.c_str(), 0, &face)) {
        std::cerr << "Error::FreeType: Failed to load font" << std::endl;
        return -1;
    } else {
        //set size to load glyph as
        FT_Set_Pixel_Sizes(face, 0, 48);
        //disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        //load first 128 characters of ASCII set
        for (unsigned char c = 0; c < 128; c++) {
            //load character glyph
            if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
                std::cerr << "Error:FreeType: Failed to load Glyph" << std::endl;
                continue;
            }

            //generate texture
            GLuint texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows,
                                        0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);

            //set texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            //new store character for later use
            Character character = {
                texture,
                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                static_cast<unsigned int>(face->glyph->advance.x)
            };

            Characters.insert(std::pair<char, Character>(c, character));
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    //destory FreeType once we're finished
    FT_Done_Face(face);
    FT_Done_FreeType(ft);


    //configuree VAO/VBO for texture quads
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //render loop
    while (!glfwWindowShouldClose(window)){
        //input
        processInput(window);

        //render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        renderText(shader, "This is sample text", 25.0f, 25.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
        renderText(shader, "(C) LearnOpenGL.com", 540.0f, 570.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));

        //glfw: swap buffer and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}


//query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

//render line of text
void renderText(Shader &shader, std::string text, float x, float y, float scale, glm::vec3 color)
{
    //active corresponding render state
    shader.use();
    glUniform3f(glGetUniformLocation(shader.Id, "textColor"), color.x , color.y, color.z);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(VAO);

    for (std::string::const_iterator c = text.begin(); c != text.end(); c++) {
        Character ch = Characters[*c];

        float xpos = x + ch.bearing.x * scale;
        float ypos = y - (ch.size.y - ch.bearing.y) * scale;

        float w = ch.size.x * scale;
        float h = ch.size.y * scale;

        //update VBO for each character
        float vertices[6][4] = {
            // glyph coordinate     texture
            {xpos,      ypos + h,   0.0f, 0.0f},
            {xpos,      ypos,       0.0f, 1.0f},
            {xpos + w,  ypos,       1.0f, 1.0f},

            {xpos,      ypos + h,   0.0f, 0.0f},
            {xpos + w,  ypos,       1.0f, 1.0f},
            {xpos + w,  ypos + h,   1.0f, 0.0f}
        };

        //render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.textureId);
        //update content of VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER,0 , sizeof(vertices), vertices); /// be sure to use glBufferSubData and not glBufferData

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        //render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        //noew advance cursors for next glyph(note that advance is number of 1/64 pixels)
        x += (ch.advance >> 6) * scale; // bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
