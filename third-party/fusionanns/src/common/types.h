#pragma once
#include <cstdint> // for uint32_t, uint16_t

// 定义SSD的页大小 (通常是4KB)
constexpr int SSD_PAGE_SIZE = 4096;

// 存储每个向量在 packed 文件中的位置
struct VectorLocation {
  uint32_t page_id;        // 所在的页ID
  uint16_t offset_in_page; // 在该页内的偏移量
                           // (uint16_t 足够表示 0-4095 的偏移)
};
