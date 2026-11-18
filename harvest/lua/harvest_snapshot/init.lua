-- Evaluator-only observations; the installed game components/rules are reused.
local tensor = require 'system.tensor'
local root = 'meltingpot.lua.modules.'
require(root .. 'component_library')
require(root .. 'avatar_library')
require 'meltingpot.lua.levels.allelopathic_harvest.components'

local api = require(root .. 'api_factory').apiFactory{
  Simulation = require(root .. 'base_simulation').BaseSimulation,
  settings = {spriteSize = 8, maxEpisodeLengthFrames = 2000, simulation = {}},
}
local originalInit = api.init
function api:init(kwargs)
  originalInit(self, kwargs)
  local observations = self._observations
  observations[#observations + 1] = {
    name = 'WORLD.BERRY_SNAPSHOT', type = 'tensor.Int32Tensor', shape = {348, 4},
    func = function(grid)
      local result = tensor.Int32Tensor(348, 4):fill(0)
      local index = 0
      for _, object in ipairs(self.simulation:getAllGameObjects()) do
        if object:hasComponent('Berry') then
          index = index + 1
          local p = object:getPosition()
          local berry = object:getComponent('Berry')
          result(index, 1):fill(p[1])
          result(index, 2):fill(p[2])
          result(index, 3):fill(berry:getBerryColorId())
          result(index, 4):fill(berry:isRipe() and 1 or 0)
        end
      end
      assert(index == 348, 'This observer supports the default 348-patch open map')
      return result
    end,
  }
  observations[#observations + 1] = {
    name = 'WORLD.AVATAR_SNAPSHOT', type = 'tensor.Int32Tensor', shape = {16, 5},
    func = function(grid)
      local result = tensor.Int32Tensor(16, 5):fill(-1)
      local orientation = {N = 0, E = 1, S = 2, W = 3}
      for index = 1, 16 do
        local object = self.simulation:getAvatarFromIndex(index)
        local alive = object:getComponent('Avatar'):isAlive()
        result(index, 4):fill(alive and 1 or 0)
        if alive then
          local p = object:getPosition()
          result(index, 1):fill(p[1])
          result(index, 2):fill(p[2])
          result(index, 3):fill(orientation[object:getOrientation()])
          result(index, 5):fill(object:getComponent('ColorZapper').colorId)
        end
      end
      return result
    end,
  }
end
return api
