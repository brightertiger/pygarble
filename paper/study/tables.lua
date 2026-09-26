-- Render Pandoc tables as ordinary LaTeX floats, compatible with two columns.
local function latex(blocks)
  return (pandoc.write(pandoc.Pandoc(blocks), 'latex'):gsub('%s+$', ''))
end

function Table(t)
  local columns = #t.colspecs
  local environment = columns >= 5 and 'table*' or 'table'
  local lines = {'\\begin{' .. environment .. '}[htbp]', '\\centering', '\\small'}
  if #t.caption.long > 0 then
    table.insert(lines, '\\caption{' .. latex(t.caption.long) .. '}')
  end
  table.insert(lines, '\\begin{tabular}{@{}l' .. string.rep('r', columns - 1) .. '@{}}')
  table.insert(lines, '\\toprule')
  local function row(r)
    local cells = {}
    for _, c in ipairs(r.cells) do
      table.insert(cells, latex(c.contents))
    end
    table.insert(lines, table.concat(cells, ' & ') .. ' \\\\')
  end
  for _, r in ipairs(t.head.rows) do row(r) end
  table.insert(lines, '\\midrule')
  for _, body in ipairs(t.bodies) do
    for _, r in ipairs(body.head) do row(r) end
    for _, r in ipairs(body.body) do row(r) end
  end
  for _, r in ipairs(t.foot.rows) do row(r) end
  table.insert(lines, '\\bottomrule')
  table.insert(lines, '\\end{tabular}')
  table.insert(lines, '\\end{' .. environment .. '}')
  return pandoc.RawBlock('latex', table.concat(lines, '\n'))
end
