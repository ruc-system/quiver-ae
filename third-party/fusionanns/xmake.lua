set_project("fusionanns")

set_languages("cxx17")
add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_strip("none")

package("SPTAG")
    add_deps("cmake")
    set_sourcedir(path.join(os.scriptdir(), "extern", "SPTAG"))

    on_install(function (package)
        import("package.tools.cmake")

        -- 最小必要 CMake 选项
        -- -DTBB=OFF: 禁用 TBB 依赖，避免需要 FindTBB.cmake
        -- -DLIBRARYONLY=ON: 只构建核心库，跳过 Python wrapper / 测试等
        local configs = {
            "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"),
            "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"),
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_MODULE_PATH=/usr/local/cuda/lib64/cmake/thrust",
            "-DTBB=OFF",
            "-DLIBRARYONLY=ON"
        }

        -- 仅构建（SPTAG 自己会把产物放到源码根的 Release/）
        -- 使用 try-catch：Python wrapper 可能因缺少 .py 文件而失败，但核心库已编译成功
        try { function() cmake.build(package, configs) end, catch { function(e) print("SPTAG build had non-fatal errors (likely Python wrappers), continuing...") end }}

        -- 校验核心库是否存在
        local releasedir = path.join(package:sourcedir(), "Release")
        assert(os.isfile(path.join(releasedir, "libSPTAGLib.so")) or os.isfile(path.join(releasedir, "libSPTAGLibStatic.a")),
               "SPTAG core library not found after build!")

        -- 安装头文件：保持 inc/ 前缀，便于 #include <inc/...>
        os.cp(path.join(package:sourcedir(), "AnnService", "inc"),
              path.join(package:installdir("include")))

        -- 安装库文件：从源码根的 Release/ 拷贝到包前缀
        local releasedir = path.join(package:sourcedir(), "Release")
        os.trycp(path.join(releasedir, "*.a"),      package:installdir("lib"))
        os.trycp(path.join(releasedir, "*.so*"),    package:installdir("lib"))
        os.trycp(path.join(releasedir, "*.dylib*"), package:installdir("lib"))
        os.trycp(path.join(releasedir, "*.lib"),    package:installdir("lib"))
        os.trycp(path.join(releasedir, "*.dll"),    package:installdir("bin"))

        -- 通知用包方：包含/库路径与链接名
        package:add("includedirs", "include")
        package:add("linkdirs",    "lib")
        package:add("links",       "SPTAGLib")
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets([[
            #include <inc/Core/Common.h>
            void test() {}
        ]], {configs = {languages = "c++17"}}))
    end)
package_end()

-- Vendored CPU FAISS (same layout as extern/SPTAG). Not a system / AE-side prefix.
package("faiss")
    add_deps("cmake", "openmp")
    add_deps("openblas", { system = true })
    set_sourcedir(path.join(os.scriptdir(), "extern", "faiss"))
    if is_plat("linux") then
        add_syslinks("pthread")
    end

    on_install(function (package)
        import("package.tools.cmake")
        local configs = {
            "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"),
            "-DBUILD_SHARED_LIBS=OFF",
            "-DBUILD_TESTING=OFF",
            "-DFAISS_ENABLE_GPU=OFF",
            "-DFAISS_ENABLE_PYTHON=OFF",
            "-DFAISS_ENABLE_C_API=OFF",
            "-DFAISS_ENABLE_EXTRAS=OFF",
        }
        cmake.install(package, configs)
        package:add("links", "faiss")
    end)

    on_test(function (package)
        assert(package:has_cxxtypes("faiss::MultiIndexQuantizer", {
            configs = {languages = "c++17"},
            includes = "faiss/IndexPQ.h"
        }))
    end)
package_end()

package("cccl")
    set_kind("library", {headeronly = true})
    add_urls("https://github.com/NVIDIA/cccl/releases/download/v$(version)/cccl-v$(version).tar.gz")
    add_versions("3.1.0", "0d051a49b9f75776a9fc34bc503d8670daeaf3f42b295ea0c6c8cb86f4fe4050")
    on_install(function (package)
        os.cp("include/*", package:installdir("include"))
        os.cp("lib/*", package:installdir("lib"))
        package:add("includedirs", "include")
    end)
package_end()

package("cutlass")
    set_kind("library", {headeronly = true})
    add_urls("https://github.com/NVIDIA/cutlass/archive/refs/tags/v$(version).tar.gz")
    add_versions("4.3.2", "e84ccd6b0c749ca87a845fb57df3d8898257bda404c5bc93ad0fb51d6decf54d")
    on_install(function (package)
        os.cp("include/*", package:installdir("include"))
        os.cp("lib/*", package:installdir("lib"))
        package:add("includedirs", "include")
    end)
package_end()

package("cuco")
    set_kind("library", {headeronly = true})
    add_deps("cccl", "cmake")
    add_urls("https://github.com/NVIDIA/cuCollections.git")
    on_install(function (package)
        local cccl = package:dep("cccl")
        local configs = {
            "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"),
            "-DBUILD_TESTS=OFF",
            "-DBUILD_BENCHMARKS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DINSTALL_CUCO=ON",
            "-DCMAKE_PREFIX_PATH=" .. cccl:installdir(),
            "-DCCCL_ROOT=" .. cccl:installdir(),
        }
        import("package.tools.cmake").install(package, configs)
    end)
    on_test(function (package)
        assert(os.exists(path.join(package:installdir(), "include/cuco/static_map.cuh")))
    end)
package_end()

-- libcuvs: 通过 conda 安装的预编译库（推荐方式）
-- 安装命令: conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.8
-- 默认安装路径: /data/miniconda（可通过 --libcuvs_root=<path> 指定）

option("libcuvs_root")
    set_default("/data/miniconda")
    set_showmenu(true)
    set_description("Path to conda prefix where libcuvs is installed (e.g. /data/miniconda)")
option_end()

-- 获取 libcuvs conda 安装路径
local function libcuvs_root()
    return get_config("libcuvs_root") or "/data/miniconda"
end

-- 检测 conda 安装的 libcuvs 是否可用
local function find_cuvs_conda()
    local prefix = libcuvs_root()
    if not prefix or prefix == "" then return false end
    local has_lib = os.isfile(path.join(prefix, "lib", "libcuvs.so"))
    local has_headers = os.isdir(path.join(prefix, "include", "cuvs"))
    return has_lib and has_headers
end

-- 项目级 define：conda 中的 libcuvs 存在时添加宏定义
if find_cuvs_conda() then
    add_defines("FUSIONANNS_USE_CUVS_SELECT_K", "FUSIONANNS_USE_CUVS_CAGRA",
                "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
    print("  [libcuvs] Found libcuvs at " .. libcuvs_root() .. " (GPU TopK + CAGRA enabled)")
end

-- 依赖：faiss / SPTAG 来自 extern/；BLAS/Boost 仍用系统包
add_requires("faiss")
add_requires("openblas", { system = true })
add_requires("boost", { system = true })
add_requires("openmp", "cli11", "SPTAG", "cccl", "cutlass")

set_policy("build.ccache", true)

-- 统一配置你机器的 CUDA 架构：
--   - "sm_89" 生成 SASS（本机架构）
--   - "compute_89" 额外生成 PTX 以提升兼容性
--   如需自适配，可改成 "native"（自动检测本机 GPU）
local cuda_gencodes = { "sm_80", "compute_80" }

-- ===== 静态库：包含 CPU 与 CUDA 源码 =====
target("fusionanns_lib")
    set_kind("static")
    -- 让该目标用 CUDA 工具链（NVCC 编译 .cu；cc/cxx 仍用宿主编译器）
    set_toolchains("cuda")
    add_links("uring", "cublas")
    -- 源码：既有 .cpp 又有 .cu
    add_files("src/**/*.cpp", "src/**/*.cu")
    -- 头文件对依赖该库的可执行目标可见
    add_includedirs("src", { public = true })
    add_packages("SPTAG", "faiss", "openblas", "boost", "cccl", "cutlass")
    add_packages("openmp")

    -- CUDA 编译与架构
    add_cuflags("-std=c++17", "--expt-extended-lambda", "--default-stream per-thread")
    add_cugencodes(table.unpack(cuda_gencodes))

    -- 关键：静态库默认不会做 device-link。
    -- 若最终二进制没有 .cu，则必须在静态库上启用 devlink，否则会有 device 符号未定义。
    set_policy("build.cuda.devlink", true)

    add_cxxflags("-fopenmp", {force = true})
    add_ldflags("-fopenmp", {force = true})
    
    on_load(function (target)
        local pkg = target:pkg("cccl")
        if pkg then
            local inc = path.join(pkg:installdir(), "include")
            local existing = os.getenv("NVCC_PREPEND_FLAGS") or ""
            local injected = "-I" .. inc
            if existing ~= "" then
                injected = existing .. " " .. injected
            end
            os.setenv("NVCC_PREPEND_FLAGS", injected)
        end
        -- cuVS 从 conda 安装：添加 include 和 link 路径
        local prefix = libcuvs_root()
        if prefix and os.isfile(path.join(prefix, "lib", "libcuvs.so")) then
            local cuvs_inc = path.join(prefix, "include")
            local cuvs_lib = path.join(prefix, "lib")
            -- CCCL 头文件（conda 的 cuda-cccl 包安装到 targets/x86_64-linux/include/）
            local cccl_inc = path.join(prefix, "targets", "x86_64-linux", "include")
            -- RAPIDS 扩展 CCCL 头文件（librmm 安装到 include/rapids/，包含 cuda::mr::any_resource 等）
            local rapids_inc = path.join(prefix, "include", "rapids")
            -- 头文件（cuvs, raft, rmm 均在 conda prefix/include/ 下）
            target:add("includedirs", cuvs_inc)
            if os.isdir(cccl_inc) then target:add("includedirs", cccl_inc) end
            if os.isdir(rapids_inc) then target:add("includedirs", rapids_inc) end
            -- NVCC 也需要 cuVS/raft/rmm/CCCL 头文件路径
            local nvcc_flags = os.getenv("NVCC_PREPEND_FLAGS") or ""
            local cuvs_nvcc = "-I" .. cuvs_inc
            if os.isdir(rapids_inc) then cuvs_nvcc = cuvs_nvcc .. " -I" .. rapids_inc end
            if os.isdir(cccl_inc) then cuvs_nvcc = cuvs_nvcc .. " -I" .. cccl_inc end
            if nvcc_flags ~= "" then
                cuvs_nvcc = cuvs_nvcc .. " " .. nvcc_flags
            end
            os.setenv("NVCC_PREPEND_FLAGS", cuvs_nvcc)
            -- 链接
            target:add("linkdirs", cuvs_lib)
            target:add("links", "cuvs")
        end
    end)

-- ===== 索引构建应用 =====
target("build_index")
    set_kind("binary")
    add_files("app/build_index.cpp")
    add_deps("fusionanns_lib")
    add_includedirs("/usr/local/cuda/include")
    add_packages("SPTAG", "faiss", "openblas", "boost", "cli11")
    add_packages("openmp")
    on_load(function (target)
        local prefix = libcuvs_root()
        if prefix and os.isfile(path.join(prefix, "lib", "libcuvs.so")) then
            local cuvs_lib = path.join(prefix, "lib")
            target:add("linkdirs", cuvs_lib)
            target:add("rpathdirs", cuvs_lib)
            target:add("links", "cublas", "cusolver", "cuvs", "rmm")
        end
    end)
    set_targetdir("bin")
    set_rundir("$(projectdir)")
    add_linkdirs("extern/SPTAG/Release")
    add_rpathdirs(path.join(os.projectdir(), "extern/SPTAG/Release"))
    add_ldflags("-fopenmp", {force = true})

-- ===== CAGRA 图构建工具 =====
target("build_cagra_graph")
    set_kind("binary")
    add_files("app/build_cagra_graph.cpp")
    add_deps("fusionanns_lib")
    add_includedirs("/usr/local/cuda/include")
    add_packages("cli11")
    add_packages("openmp")
    on_load(function (target)
        local prefix = libcuvs_root()
        if prefix and os.isfile(path.join(prefix, "lib", "libcuvs.so")) then
            local cuvs_lib = path.join(prefix, "lib")
            target:add("defines", "FUSIONANNS_USE_CUVS_CAGRA")
            target:add("linkdirs", cuvs_lib)
            target:add("rpathdirs", cuvs_lib)
            target:add("links", "cublas", "cusolver", "cuvs", "rmm")
        end
    end)
    set_targetdir("bin")
    set_rundir("$(projectdir)")
    add_linkdirs("extern/SPTAG/Release")
    add_rpathdirs(path.join(os.projectdir(), "extern/SPTAG/Release"))
    add_ldflags("-fopenmp", {force = true})

-- ===== 查询服务应用 =====
target("query_server")
    set_kind("binary")
    add_files("app/query_server.cpp")
    add_deps("fusionanns_lib")
    add_includedirs("/usr/local/cuda/include")
    add_packages("openmp")
    add_packages("SPTAG", "faiss", "openblas", "boost", "cli11", "cutlass")
    add_links("uring", "dl")
    on_load(function (target)
        local prefix = libcuvs_root()
        if prefix and os.isfile(path.join(prefix, "lib", "libcuvs.so")) then
            local cuvs_lib = path.join(prefix, "lib")
            target:add("defines", "FUSIONANNS_USE_CUVS_SELECT_K",
                       "FUSIONANNS_USE_CUVS_CAGRA")
            target:add("linkdirs", cuvs_lib)
            target:add("ldflags", "-L" .. cuvs_lib, "-Wl,-rpath-link," .. cuvs_lib)
            target:add("rpathdirs", cuvs_lib)
            -- 链接顺序：cublas 需在 cusolver 之前（libcuvs 依赖链）
            target:add("links", "cublas", "cusolver", "cuvs", "rmm")
        end
    end)
    set_targetdir("bin")
    set_rundir("$(projectdir)")
    add_linkdirs("extern/SPTAG/Release")
    add_rpathdirs(path.join(os.projectdir(), "extern/SPTAG/Release"))
    -- set_policy("build.sanitizer.address", true)
    add_ldflags("-fopenmp", {force = true})

-- ===== SPTAG 聚类中心图独立构建工具 =====
target("build_centroid_graph_standalone")
    set_kind("binary")
    add_files("app/build_centroid_graph_standalone.cpp")
    add_deps("fusionanns_lib")
    add_includedirs("/usr/local/cuda/include")
    add_packages("SPTAG", "faiss", "openblas", "boost", "cli11")
    add_packages("openmp")
    on_load(function (target)
        local prefix = libcuvs_root()
        if prefix and os.isfile(path.join(prefix, "lib", "libcuvs.so")) then
            local cuvs_lib = path.join(prefix, "lib")
            target:add("linkdirs", cuvs_lib)
            target:add("rpathdirs", cuvs_lib)
            target:add("links", "cublas", "cusolver", "cuvs", "rmm")
        end
    end)
    set_targetdir("bin")
    set_rundir("$(projectdir)")
    add_linkdirs("extern/SPTAG/Release")
    add_rpathdirs(path.join(os.projectdir(), "extern/SPTAG/Release"))
    add_ldflags("-fopenmp", {force = true})

-- ===== 聚类中心 Recall 评估工具 =====
target("eval_centroid_recall")
    set_kind("binary")
    add_files("app/eval_centroid_recall.cpp")
    add_deps("fusionanns_lib")
    add_includedirs("/usr/local/cuda/include")
    add_packages("SPTAG", "faiss", "openblas", "boost", "cli11")
    add_packages("openmp")
    on_load(function (target)
        local prefix = libcuvs_root()
        if prefix and os.isfile(path.join(prefix, "lib", "libcuvs.so")) then
            local cuvs_lib = path.join(prefix, "lib")
            target:add("linkdirs", cuvs_lib)
            target:add("rpathdirs", cuvs_lib)
            target:add("links", "cublas", "cusolver", "cuvs", "rmm")
        end
    end)
    set_targetdir("bin")
    set_rundir("$(projectdir)")
    add_linkdirs("extern/SPTAG/Release")
    add_rpathdirs(path.join(os.projectdir(), "extern/SPTAG/Release"))
    add_ldflags("-fopenmp", {force = true})

-- ===== 索引验证工具 =====
target("index_checker")
    set_kind("binary")
    add_files("app/index_checker.cpp", "src/common/datasource.cpp")
    add_packages("faiss", "cli11")
    add_includedirs("src")
    set_targetdir("bin")
    set_rundir("$(projectdir)")
