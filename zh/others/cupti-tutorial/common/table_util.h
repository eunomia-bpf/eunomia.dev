#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>

enum class OverflowMode { WrapHard, WrapWord };
enum class Alignment { Left, Right };

class Table
{
public:
    struct Column
    {
        std::string header;
        int width;
        Alignment alignment;
        OverflowMode overflow;
    };

    Table(std::vector<Column> columns)
        : m_columns(std::move(columns))
    {}

    void addRow(const std::vector<std::string>& row) {
        m_rows.push_back(row);
    }

    void print() const
    {
        printSeparator(std::cout);
        printHeaders(std::cout);
        printSeparator(std::cout);

        for (const auto& row : m_rows)
        {
            printRow(row, std::cout);
            printSeparator(std::cout);
        }
        std::cout << "Table printed to console." << std::endl;
    }

    void exportToCSV(const std::string& filename) const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        for (size_t i = 0; i < m_columns.size(); ++i)
        {
            if (!m_columns[i].width) {
                continue;
            }
            file << m_columns[i].header;
            if (i < m_columns.size() - 1) {
                file << ",";
            }
        }

        file << "\n";
        for (const auto& row : m_rows)
        {
            for (size_t i = 0; i < row.size(); ++i)
            {
                if (!m_columns[i].width) {
                    continue;
                }
                file << '"' << row[i] << '"';
                if (i < row.size() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
        std::cout << "Table exported to " << filename << std::endl;
    }

    void exportToTextFile(const std::string& filename) const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        printSeparator(file);
        printHeaders(file);
        printSeparator(file);

        for (const auto& row : m_rows)
        {
            printRow(row, file);
            printSeparator(file);
        }
        file.close();
        std::cout << "Table exported to " << filename << std::endl;
    }

private:
    std::vector<Column> m_columns;
    std::vector<std::vector<std::string>> m_rows;

    void printSeparator(std::ostream& os) const
    {
        for (const auto& col : m_columns)
        {
            if (col.width) {
                os << "+" << std::string(col.width, '-');
            }
        }
        os << "+" << std::endl;
    }

    void printHeaders(std::ostream& os) const
    {
        for (const auto& col : m_columns)
        {
            if (col.width) {
                printAlignedText(col.header, col.width, col.alignment, os);
            }
        }
        os << "|" << std::endl;
    }

    void printRow(const std::vector<std::string>& row, std::ostream& os) const
    {
        constexpr int padding = 2;
        size_t maxLines = 0;
        std::vector<std::vector<std::string>> wrappedText;
        for (size_t i = 0; i < m_columns.size(); ++i)
        {
            wrappedText.push_back(wrapText(row[i], m_columns[i].width - padding, m_columns[i].overflow));
            maxLines = std::max(maxLines, wrappedText.back().size());
        }

        for (size_t i = 0; i < maxLines; ++i)
        {
            for (size_t j = 0; j < m_columns.size(); ++j)
            {
                if (m_columns[j].width)
                {
                    std::string text = (i < wrappedText[j].size()) ? wrappedText[j][i] : "";
                    printAlignedText(text, m_columns[j].width, m_columns[j].alignment, os);
                }
            }
            os << "|" << std::endl;
        }
    }

    std::vector<std::string> wrapText(const std::string& text, int width, OverflowMode overflow) const
    {
        std::vector<std::string> lines;
        if (overflow == OverflowMode::WrapHard)
        {
            for (size_t i = 0; i < text.length(); i += width) {
                lines.push_back(text.substr(i, width));
            }
        }
        else
        {
            std::istringstream iss(text);
            std::string word, line;
            while (iss >> word)
            {
                if (line.length() + word.length() + 1 > static_cast<size_t>(width))
                {
                    lines.push_back(line);
                    line = word;
                }
                else
                {
                    if (!line.empty()) line += " ";
                    line += word;
                }
            }

            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        return lines;
    }

    void printAlignedText(const std::string& text, int width, Alignment align, std::ostream& os) const
    {
        if (width)
        {
            os << "| ";
            if (align == Alignment::Left) {
                os << std::left << std::setw(width - 1) << text;
            } else {
                os << std::setw(width - 1) << text;
            }
        }
    }
};

