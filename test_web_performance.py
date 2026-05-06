"""测试 Web API 性能"""
import asyncio
import time
import aiohttp

BASE_URL = "http://127.0.0.1:8000"


async def test_health():
    """测试健康检查端点"""
    print("1. Testing /health ...")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as resp:
            data = await resp.json()
            elapsed = time.time() - start
            print(f"   [PASS] /health: {elapsed:.3f}s, response: {data}")
            return elapsed


async def test_create_job():
    """测试创建任务（不实际运行）"""
    print("\n2. Testing POST /jobs (create job) ...")
    payload = {
        "article": "这是一个测试文章。人工智能正在改变世界。",
        "source_lang": "zh",
        "translate_to": None,
        "aspect_ratio": "9:16",
        "nlp_backend": "local",
        "voice_primary": "zh-CN-XiaoxiaoNeural",
        "voice_secondary": None,
        "bgm_enabled": False,
        "burn_subtitles": True,
    }
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/jobs", json=payload) as resp:
            data = await resp.json()
            elapsed = time.time() - start
            print(f"   [PASS] POST /jobs: {elapsed:.3f}s, job_id: {data.get('job_id', 'N/A')}")
            return data.get("job_id")


async def test_list_jobs():
    """测试列出任务"""
    print("\n3. Testing GET /jobs (list jobs) ...")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/jobs") as resp:
            data = await resp.json()
            elapsed = time.time() - start
            job_count = len(data.get("jobs", []))
            print(f"   [PASS] GET /jobs: {elapsed:.3f}s, {job_count} jobs")
            return elapsed


async def test_job_status(job_id):
    """测试获取任务状态"""
    print(f"\n4. Testing GET /jobs/{job_id} (get job status) ...")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/jobs/{job_id}") as resp:
            data = await resp.json()
            elapsed = time.time() - start
            print(f"   [PASS] GET /jobs/{job_id}: {elapsed:.3f}s, status: {data.get('status')}")
            return elapsed


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Web Performance Test")
    print("=" * 60)

    try:
        # Test 1: Health check
        await test_health()

        # Test 2: Create job
        job_id = await test_create_job()
        
        # Test 3: List jobs
        await test_list_jobs()

        # Test 4: Get job status
        if job_id:
            await test_job_status(job_id)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except aiohttp.ClientConnectorError:
        print("\n[FAIL] Cannot connect to server. Is it running?")
        print("Start with: uv run uvicorn app.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
