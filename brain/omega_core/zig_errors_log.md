# Журнал нетиповых ошибок Zig (версия 0.16.0)

В этом файле собираются специфические ошибки, связанные с переходом на версию Zig 0.16.0 и выше в рамках сборки `OmegaCore`. 

## 1. Проблема с инициализацией ArrayList
- **Ошибка**: `error: struct 'array_list.Aligned([]const u8,null)' has no member named 'init'`
- **Причина**: Использование `std.ArrayList([]const u8).init(b.allocator)` ломается, так как внутреннее устройство и методы инициализации `ArrayList` изменились.
- **Решение**: Полностью отказаться от `ArrayList` при формировании флагов конфигурации (`cxx_flags`). Вместо этого выделять память напрямую через `b.allocator.alloc([]const u8, len)` и заполнять слайс вручную по индексам.

## 2. Проблема с очисткой папок (FileSystem API)
- **Ошибка**: Указание на `std.fs.cwd().deleteTree("zig-out") catch {};` (нет метода или структуры).
- **Причина**: В новом API модуля `std.fs` метод `deleteTree` напрямую у `cwd()` либо удалён, либо изменена его сигнатура (требуются абсолютные пути или другой механизм итерации).
- **Решение**: Не производить манипуляций с файловой системой (особенно удаление директорий кеша `.zig-cache`, `zig-out`) внутри самого скрипта `build.zig`. Любую предварительную очистку следует реализовать в обёрточном скрипте (например, в PowerShell `.ps1` через `Remove-Item` или использованием уникальных `--cache-dir`), а сборщику Zig оставить только компиляцию артефактов.

## 3. Проблема PyBind11 и Thread Local на Windows
- **Ошибка**: `error: 'c10::impl::raw_local_dispatch_key_set' cannot be thread local when declared 'dllimport'`
- **Причина**: Использование `thread_local` переменных в статических/динамических модулях под Windows при связке с Python.
- **Решение**: Обязательная передача флага `-DCOMPILING_DLL` в параметры `build.zig` (для C++ флагов) на платформе Windows, а также точное указание `-DTORCH_EXTENSION_NAME=имя_вашего_модуля`.

## 4. Ошибка AccessDenied при сборке MinGW CRT
- **Ошибка**: `error: sub-compilation of mingw-w64 libmingw32.lib failed`, `clang exited with code 5`, `AccessDenied` в файлах `lib/libc/mingw/crt/...`.
- **Причина**: При таргете `x86_64-windows-gnu` компилятор Zig (использующий внутри clang) пытается собрать базовые библиотеки C (libc/mingw). На некоторых конфигурациях Windows встроенный антивирус (Windows Defender) или права доступа (особенно в AppData) блокируют прозрачную компиляцию системных библиотек, выдавая AccessDenied.
- **Решение**: Во-первых, при работе в Python-окружении (особенно с PyTorch) на Windows рекомендуется собирать с таргетом MSVC (ABI Windows), а не GNU. Изменить таргет сборки на `-Dtarget=x86_64-windows-msvc` в PowerShell скриптах вызова сборки.

## 5. Ошибки путей включения (Include Paths) для PyTorch C++ API
- **Ошибка**: `error: in file included from ... torch/csrc/api/include/torch/all.h:8: #include <torch/autograd.h> file not found`.
- **Причина**: Неправильно указана глубина пути для `torch/csrc/api/include`. Файлы внутри `torch/` ищутся относительно базовой директории API. Если прописать путь прямо до `include/torch/csrc/api/include`, то `#include <torch/...>` будет искать `include/torch/csrc/api/include/torch/...`, что верно, но иногда PyTorch требует корень самого `site-packages` или просто `include`.
- **Решение**: Убедиться, что в `build.zig` добавлены не только `include` и `include/torch/csrc/api/include`, но и путь к корневому пакету Python, чтобы маппинг модулей C10 и Torch работал корректно. Часто достаточно `-I <python_dir>/Lib/site-packages/torch/include` и `-I <python_dir>/Lib/site-packages/torch/include/torch/csrc/api/include`.

## 6. Ошибки __declspec и libcxxabi (MSVC Runtime) на Windows
- **Ошибка**: `error: '__declspec' attributes are not enabled; use '-fdeclspec' or '-fms-extensions'`, а также `error: sub-compilation of libcxxabi failed` и `target of using declaration conflicts`.
- **Причина**: В оптимизированных флагах C++ (`cxx_flags`) по умолчанию стоял флаг `-fno-declspec` для отключения нестандартных спецификаторов. Однако заголовки Windows (и MSVC CRT) критически зависят от `__declspec(dllexport)`, `__declspec(noreturn)` и так далее. Использование `link_libcpp = true` на таргете MSVC также может приводить к конфликту встроенной в Zig LLVM libc++ с родным MSVC `vcruntime.h`.
- **Решение**: Убрать из флагов компиляции `-fno-declspec`. Добавить `-fdeclspec` и `-fms-extensions`. При линковке для ABI `msvc` избегать явного `module.linkSystemLibrary("c++", .{})`, так как это сбивает с толку внутренний Clang Zig. Достаточно оставить встроенные `module.link_libc = true` и `module.link_libcpp = true` или вообще позволить Clang самому найти нужный CRT.

## 7. Конфликт std::type_info и libcxxabi при MSVC
- **Ошибка**: `error: sub-compilation of libcxxabi failed`, `cannot define or redeclare '~type_info'` и `invalid operands to binary expression ('const std::type_info' and 'const std::type_info')`.
- **Причина**: Когда для таргета `x86_64-windows-msvc` в Zig устанавливается флаг `module.link_libcpp = true`, компилятор Zig пытается собрать свою собственную (основанную на LLVM) версию `libcxxabi` и стандартную библиотеку C++. Это вызывает тяжелые конфликты типов и пространств имен (особенно `std::type_info`) с родными заголовками MSVC (например, `vcruntime_typeinfo.h`), которые неявно подключаются при компиляции через цепочки инклудов Windows SDK/PyTorch.
- **Решение**: Строго отключать `module.link_libcpp = true` в `build.zig`, если `target.result.abi == .msvc`. Оставляем только `module.link_libc = true`. Родной компилятор/линкер MSVC сам подтянет необходимые зависимости C++ Runtime (libcpmt/msvcprt) для Windows.

## 8. Ошибки _mm256_load_si256 и std::numeric_limits::infinity (AVX/Fast-Math)
- **Ошибка**: `always_inline function '_mm256_load_si256' requires target feature 'avx'` и `use of infinity is undefined behavior due to the currently enabled floating-point options`.
- **Причина**: В оптимизированных флагах C++ (`cxx_flags`) присутствовал флаг `-ffast-math`. Этот флаг нарушает стандарт IEEE-754 и полностью отключает поддержку `NaN` и бесконечностей (`infinity(0)`). Заголовки PyTorch (например, `clip_grad.h`) активно используют `std::numeric_limits<double>::infinity()`, что приводит к ошибке при включенном `-ffast-math`. Кроме того, исходный код использует интринсики AVX2 (`__m256i`, `_mm256_load_si256`), но компилятор не знал об этом, так как флаги архитектуры не были переданы.
- **Решение**: Убрать из флагов компиляции в `build.zig` флаг `-ffast-math`. Вместо него явно добавить флаги набора команд процессора: `-mavx2` и `-mfma`. Это разрешит использование интринсиков и восстановит совместимость с математикой LibTorch.

## 9. Ошибка 'dllimport' на enum class в заголовках PyTorch (TORCH_API)
- **Ошибка**: `error: 'dllimport' attribute only applies to functions, variables, classes, and Objective-C interfaces` на строках типа `enum class TORCH_API Float32MatmulPrecision`.
- **Причина**: В заголовках PyTorch/ATen на Windows макрос `TORCH_API` раскрывается в `__declspec(dllimport)`. Компилятор MSVC исторически позволяет применять `__declspec` к перечислениям (`enum class`), но строгий парсер Clang в Zig (особенно если включен флаг `-fdeclspec`) отклоняет это как нарушение стандарта C++.
- **Решение**: Убрать флаг `-fdeclspec` из массива `cxx_flags`. Оставить только `-fms-extensions` и обязательно добавить макрос `-D_WIN32`. Это переведет парсер Clang в режим максимальной совместимости с MSVC-специфичными расширениями, заставив его игнорировать некорректно примененный `__declspec` на `enum class`.

## 10. Ошибка линковки LLD: bad file type (DLL instead of import library)
- **Ошибка**: `error: lld-link: ..\..\venv\Lib\site-packages\torch\lib\c10.dll: bad file type. Did you specify a DLL instead of an import library?`
- **Причина**: При использовании стандартного метода `module.linkSystemLibrary("c10", .{})` компилятор Zig делегирует поиск библиотеки линкеру `lld-link`. На Windows линкер иногда ошибочно находит `.dll` файл в системных путях (или в `LibraryPath`) раньше, чем нужную ему библиотеку импорта `.lib`. LLD не умеет линковать DLL напрямую, ему нужен именно импорт-файл `.lib`.
- **Решение**: На таргете Windows отказаться от `linkSystemLibrary` для библиотек PyTorch и Python. Вместо этого вручную собирать абсолютные или относительные пути к файлам `.lib` и отдавать их как объектные файлы через `module.addObjectFile(.{ .cwd_relative = "путь/до/lib/c10.lib" })`. Это жестко привязывает линковщик именно к правильным файлам экспорта.

## 11. Ошибка чувствительности к регистру путей (non-portable path)
- **Ошибка**: `error: non-portable path to file '<python.h>'; specified path differs in case from file name on disk`
- **Причина**: В библиотеке PyTorch в файле `python_headers.h` прописано `#include <Python.h>` (с заглавной), а на диске файл лежит как `python.h` (строчными). Clang в составе Zig под Windows работает в режиме проверки регистра (строгий case-sensitivity) портируемых путей, и считает такое несовпадением.
- **Решение**: Добавить флаг `-Wno-nonportable-include-path` в параметры компиляции `build_omega.ps1` (для `zig c++`) и в секцию `cxx_flags` файла `build.zig`. Этот флаг подавляет предупреждения о несовпадении букв на case-insensitive файловых системах.

## 12. Отсутствующие заголовки Python (Python.h / frameobject.h)
- **Ошибка**: `error: 'Python.h' file not found with <angled> include` и `error: 'frameobject.h' file not found`
- **Причина**: Заголовки исходного кода C++ PyTorch (C10/ATen/Torch) ссылаются на структуры Python. Если при вызове внешней компиляции через `zig c++` явно не передан путь к папке `include` самого Python, компилятор не сможет их найти.
- **Решение**: Динамически вычислять пути Python в обёртке `build_omega.ps1` с помощью `python -c "import sysconfig; print(sysconfig.get_path('include'))"` и передавать их компилятору через флаги `-I"$PY_INCLUDE" -L"$PY_LIB"`.

## 13. Ошибка AccessDenied на первой строке (VFS MSVC Bug)
- **Ошибка**: `omega_core.cpp:1:1: error: AccessDenied` при выполнении внутренней команды `zig build-lib ... -target x86_64-windows-msvc`
- **Причина**: В версиях Zig 0.13+ на Windows подсистема кеширования и виртуальная файловая система (VFS) для доступа к Windows SDK (MSVC) часто конфликтуют с антивирусом (Defender) или блокировками файлов (если Python держит `.pyd` в памяти). Из-за этого `build.zig` падает с `AccessDenied` не успев даже прочитать первый `#include`.
- **Решение**: Полностью отказаться от сложной системы `build.zig` и таргета `msvc`. Использовать прямой вызов компилятора: `zig c++ -shared ... -target x86_64-windows-gnu`. ABI GNU собирается напрямую без зависимостей от Windows SDK, а полученная библиотека (`.pyd` / `.dll`) всё так же абсолютно корректно загружается в PyTorch под Windows.

## 14. Ошибка DLL load failed при импорте скомпилированного MSVC .pyd в Python
- **Ошибка**: `DLL load failed while importing omega_core_lib: The specified module could not be found.` в Python (после успешной сборки Zig).
- **Причина**: Начиная с Python 3.8 под Windows, механизм загрузки DLL библиотек (C/C++ Extensions) перестал учитывать системную переменную окружения `PATH` из соображений безопасности. Поскольку `omega_core_lib.pyd` динамически линкуется с `torch.dll` и `c10.dll`, их расположение должно быть явно указано.
- **Решение**: Перед импортом нативного `.pyd` модуля (функция `importlib.util.spec_from_file_location`), необходимо найти директорию `lib` библиотеки PyTorch и добавить её в пути поиска через `os.add_dll_directory(torch_lib_dir)`.
